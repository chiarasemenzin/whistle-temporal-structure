import json
import numpy as np
import random
from typing import List, Dict, Tuple

def load_dataset(dataset_path):
    """Load the dataset from JSON."""
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    return dataset

def split_dataset_by_bout(dataset, seed=42, train_frac=0.7, val_frac=0.15):
    """
    Split dataset into train/val/test by bouts to prevent data leakage.

    Args:
        dataset: The full dataset
        seed: Random seed for reproducibility
        train_frac: Fraction of bouts for training
        val_frac: Fraction of bouts for validation

    Returns:
        train_set, val_set, test_set: Three non-overlapping datasets
    """
    # Collect all bouts
    bouts = []  # list of (recording_name, bout_name)
    for rec_name, rec_data in dataset.items():
        for bout_name in rec_data["bouts"].keys():
            bouts.append((rec_name, bout_name))

    # Shuffle with fixed seed
    random.Random(seed).shuffle(bouts)

    # Split indices
    n = len(bouts)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_bouts = set(bouts[:n_train])
    val_bouts = set(bouts[n_train:n_train+n_val])
    test_bouts = set(bouts[n_train+n_val:])

    print(f"\nDataset split:")
    print(f"  Total bouts: {n}")
    print(f"  Train bouts: {len(train_bouts)} ({len(train_bouts)/n*100:.1f}%)")
    print(f"  Val bouts: {len(val_bouts)} ({len(val_bouts)/n*100:.1f}%)")
    print(f"  Test bouts: {len(test_bouts)} ({len(test_bouts)/n*100:.1f}%)")

    def filter_dataset(dataset, keep_bouts):
        """Filter dataset to only include specified bouts."""
        sub = {}
        for rec_name, rec_data in dataset.items():
            sub_bouts = {
                bname: bdata
                for bname, bdata in rec_data["bouts"].items()
                if (rec_name, bname) in keep_bouts
            }
            if sub_bouts:
                sub[rec_name] = {
                    "id": rec_data.get("id", rec_name),
                    "bouts": sub_bouts
                }
        return sub

    train_set = filter_dataset(dataset, train_bouts)
    val_set = filter_dataset(dataset, val_bouts)
    test_set = filter_dataset(dataset, test_bouts)

    return train_set, val_set, test_set

def create_training_samples(dataset, k, min_bout_length=None):
    """
    Create training samples using a sliding window of size k.

    Args:
        dataset: The loaded dataset
        k: Context length (number of previous whistles to use as context)
        min_bout_length: Minimum bout length to consider (default: k+1)

    Returns:
        X: List of context sequences (each is a list of k embeddings)
        y: List of target labels (the label of the next whistle)
        metadata: List of dicts with bout/whistle info for each sample
    """
    if min_bout_length is None:
        min_bout_length = k + 1

    X = []  # contexts (embeddings)
    y = []  # targets (labels)
    metadata = []

    for recording_name, recording_data in dataset.items():
        for bout_name, bout_data in recording_data["bouts"].items():
            # Get all whistles in this bout
            whistle_names = sorted(bout_data.keys(),
                                  key=lambda x: int(x.split('_')[1]))
            bout_length = len(whistle_names)

            # Skip bouts that are too short
            if bout_length < min_bout_length:
                continue

            # Extract embeddings and labels for this bout
            embeddings = [bout_data[w_name]["embedding"] for w_name in whistle_names]
            labels = [bout_data[w_name]["label"] for w_name in whistle_names]

            # Slide window through the bout
            for i in range(k, bout_length):
                context = embeddings[i-k:i]  # k previous embeddings
                target_label = labels[i]  # label of next whistle

                X.append(context)
                y.append(target_label)
                metadata.append({
                    "recording": recording_name,
                    "bout": bout_name,
                    "position": i,
                    "bout_length": bout_length,
                    "whistle_name": whistle_names[i]
                })

    return np.array(X), np.array(y), metadata

def create_test_samples_shared_positions(dataset, k_values, min_bout_length=8):
    """
    Create test samples for multiple k values using shared prediction positions.

    This ensures that all k values are evaluated on exactly the same positions
    for fair comparison.

    Args:
        dataset: The loaded dataset
        k_values: List of context lengths to prepare for
        min_bout_length: Minimum bout length for test bouts (default: 8)

    Returns:
        test_data: Dict mapping k -> (X, y, metadata)
        shared_positions: List of (recording, bout, position) tuples
    """
    max_k = max(k_values)

    # First, identify all valid prediction positions
    # (positions where we can make predictions for the largest k)
    shared_positions = []

    for recording_name, recording_data in dataset.items():
        for bout_name, bout_data in recording_data["bouts"].items():
            whistle_names = sorted(bout_data.keys(),
                                  key=lambda x: int(x.split('_')[1]))
            bout_length = len(whistle_names)

            # Only use bouts that are long enough
            if bout_length < min_bout_length:
                continue

            # All positions from max_k onwards can be predicted with any k <= max_k
            for i in range(max_k, bout_length):
                shared_positions.append({
                    "recording": recording_name,
                    "bout": bout_name,
                    "position": i,
                    "bout_length": bout_length,
                    "whistle_names": whistle_names
                })

    # Now create test samples for each k using these shared positions
    test_data = {}

    for k in k_values:
        X = []
        y = []
        metadata = []

        for pos_info in shared_positions:
            recording_name = pos_info["recording"]
            bout_name = pos_info["bout"]
            position = pos_info["position"]
            whistle_names = pos_info["whistle_names"]

            bout_data = dataset[recording_name]["bouts"][bout_name]

            # Extract embeddings and labels
            embeddings = [bout_data[w_name]["embedding"] for w_name in whistle_names]
            labels = [bout_data[w_name]["label"] for w_name in whistle_names]

            # Get context of length k and target label
            context = embeddings[position-k:position]
            target_label = labels[position]

            X.append(context)
            y.append(target_label)
            metadata.append({
                "recording": recording_name,
                "bout": bout_name,
                "position": position,
                "bout_length": pos_info["bout_length"],
                "whistle_name": whistle_names[position]
            })

        test_data[k] = {
            "X": np.array(X),
            "y": np.array(y),
            "metadata": metadata
        }

    return test_data, shared_positions

def balance_by_label(X, y, metadata, seed=42):
    """
    Balance dataset by undersampling majority classes.

    Args:
        X: Array of contexts
        y: Array of labels
        metadata: List of metadata dicts
        seed: Random seed for reproducibility

    Returns:
        X_balanced, y_balanced, metadata_balanced: Balanced dataset
    """
    from collections import Counter

    # Count samples per label
    label_counts = Counter(y)
    print(f"\n  Original label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"    {label}: {count}")

    # Find minimum count (target size for each class)
    min_count = min(label_counts.values())
    print(f"\n  Balancing to {min_count} samples per class...")

    # Group indices by label
    label_to_indices = {label: [] for label in label_counts.keys()}
    for idx, label in enumerate(y):
        label_to_indices[label].append(idx)

    # Sample min_count indices from each label
    rng = np.random.RandomState(seed)
    balanced_indices = []

    for label, indices in label_to_indices.items():
        if len(indices) >= min_count:
            sampled = rng.choice(indices, size=min_count, replace=False)
        else:
            sampled = indices  # Keep all if less than min_count
        balanced_indices.extend(sampled)

    # Shuffle the balanced indices
    rng.shuffle(balanced_indices)

    # Create balanced dataset
    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]
    metadata_balanced = [metadata[i] for i in balanced_indices]

    # Print new distribution
    balanced_counts = Counter(y_balanced)
    print(f"\n  Balanced label distribution:")
    for label, count in sorted(balanced_counts.items()):
        print(f"    {label}: {count}")

    print(f"\n  Total samples: {len(y)} -> {len(y_balanced)}")

    return X_balanced, y_balanced, metadata_balanced

def save_prepared_data(output_path, data_dict):
    """Save prepared data to disk."""
    np.savez(output_path, **data_dict)
    print(f"Saved data to {output_path}")

def main():
    dataset_path = "../data/dataset.json"
    output_dir = "../data/"

    print("Loading dataset...")
    dataset = load_dataset(dataset_path)

    # Split dataset by bouts (no overlap between train/val/test)
    print("\n" + "="*60)
    print("SPLITTING DATASET BY BOUTS")
    print("="*60)
    train_set, val_set, test_set = split_dataset_by_bout(
        dataset, seed=42, train_frac=0.7, val_frac=0.15
    )

    # Define context lengths to evaluate
    k_values = [2, 3, 4, 5, 6, 7]

    print("\n" + "="*60)
    print("PREPARING TRAINING DATA")
    print("="*60)

    # Prepare training data for each k using ONLY train_set
    train_data = {}
    train_data_balanced = {}

    for k in k_values:
        print(f"\nPreparing training data for k={k}...")
        X_train, y_train, metadata_train = create_training_samples(
            train_set, k, min_bout_length=k+1
        )

        print(f"  - Created {len(X_train)} training samples")
        print(f"  - Context shape: {X_train.shape}")
        print(f"  - Target shape: {y_train.shape}")

        # Save unbalanced data
        train_data[f"k{k}_X"] = X_train
        train_data[f"k{k}_y"] = y_train

        # Balance the data
        print(f"\n  Balancing training data for k={k}...")
        X_train_bal, y_train_bal, metadata_train_bal = balance_by_label(
            X_train, y_train, metadata_train, seed=42
        )

        train_data_balanced[f"k{k}_X"] = X_train_bal
        train_data_balanced[f"k{k}_y"] = y_train_bal

    # Save both versions
    save_prepared_data(output_dir + "train_data.npz", train_data)
    save_prepared_data(output_dir + "train_data_balanced.npz", train_data_balanced)

    print("\n" + "="*60)
    print("PREPARING TEST DATA (SHARED POSITIONS)")
    print("="*60)

    # Prepare test data with shared positions using ONLY test_set
    test_data, shared_positions = create_test_samples_shared_positions(
        test_set, k_values, min_bout_length=8
    )

    print(f"\nTotal shared prediction positions: {len(shared_positions)}")

    test_data_dict = {}
    test_data_dict_balanced = {}

    for k in k_values:
        print(f"\nTest data for k={k}:")
        X_test = test_data[k]["X"]
        y_test = test_data[k]["y"]
        metadata_test = test_data[k]["metadata"]

        print(f"  - Samples: {len(X_test)}")
        print(f"  - Context shape: {X_test.shape}")
        print(f"  - Target shape: {y_test.shape}")

        # Save unbalanced test data
        test_data_dict[f"k{k}_X"] = X_test
        test_data_dict[f"k{k}_y"] = y_test

        # Balance test data
        print(f"\n  Balancing test data for k={k}...")
        X_test_bal, y_test_bal, metadata_test_bal = balance_by_label(
            X_test, y_test, metadata_test, seed=42
        )

        test_data_dict_balanced[f"k{k}_X"] = X_test_bal
        test_data_dict_balanced[f"k{k}_y"] = y_test_bal

    # Save both versions
    save_prepared_data(output_dir + "test_data.npz", test_data_dict)
    save_prepared_data(output_dir + "test_data_balanced.npz", test_data_dict_balanced)

    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"\nFiles saved:")
    print(f"  Unbalanced:")
    print(f"    - {output_dir}train_data.npz")
    print(f"    - {output_dir}test_data.npz")
    print(f"  Balanced:")
    print(f"    - {output_dir}train_data_balanced.npz")
    print(f"    - {output_dir}test_data_balanced.npz")
    print(f"\nNOTE: Train and test sets have NO overlapping bouts!")
    print(f"      Balanced versions have equal samples per label.")

if __name__ == "__main__":
    main()
