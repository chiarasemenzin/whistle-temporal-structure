import json
import numpy as np
import random
from typing import List, Dict, Tuple
from collections import Counter

def load_dataset(dataset_path):
    """Load the dataset from JSON."""
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    return dataset

def split_dataset_by_bout(dataset, seed=42, train_frac=0.7, val_frac=0.15):
    """
    Split dataset into train/val/test by bouts to prevent data leakage.
    """
    bouts = []
    for rec_name, rec_data in dataset.items():
        for bout_name in rec_data["bouts"].keys():
            bouts.append((rec_name, bout_name))

    random.Random(seed).shuffle(bouts)

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

def create_training_samples_no_persistence(dataset, k, min_bout_length=None):
    """
    Create training samples EXCLUDING cases where target label == previous label.

    Only filters when k=2 (immediate previous).
    For k>2, we still exclude if labels[i] == labels[i-1].

    Args:
        dataset: The loaded dataset
        k: Context length
        min_bout_length: Minimum bout length to consider (default: k+1)

    Returns:
        X: Context sequences
        y: Target labels
        metadata: Metadata for each sample
    """
    if min_bout_length is None:
        min_bout_length = k + 1

    X = []
    y = []
    metadata = []

    filtered_count = 0
    total_count = 0

    for recording_name, recording_data in dataset.items():
        for bout_name, bout_data in recording_data["bouts"].items():
            whistle_names = sorted(bout_data.keys(),
                                  key=lambda x: int(x.split('_')[1]))
            bout_length = len(whistle_names)

            if bout_length < min_bout_length:
                continue

            embeddings = [bout_data[w_name]["embedding"] for w_name in whistle_names]
            labels = [bout_data[w_name]["label"] for w_name in whistle_names]

            for i in range(k, bout_length):
                total_count += 1

                context = embeddings[i-k:i]
                target_label = labels[i]
                previous_label = labels[i-1]  # Immediately previous label

                # FILTER: Skip if target == immediate previous (no A A pattern)
                if target_label == previous_label:
                    filtered_count += 1
                    continue

                X.append(context)
                y.append(target_label)
                metadata.append({
                    "recording": recording_name,
                    "bout": bout_name,
                    "position": i,
                    "bout_length": bout_length,
                    "whistle_name": whistle_names[i]
                })

    print(f"  Filtered {filtered_count}/{total_count} samples with label persistence")
    print(f"  Kept {len(X)} samples ({len(X)/total_count*100:.1f}%)")

    return np.array(X), np.array(y), metadata

def create_test_samples_shared_positions_no_persistence(dataset, k_values, min_bout_length=5):
    """
    Create test samples with shared positions, excluding label persistence.
    """
    max_k = max(k_values)

    # Identify all valid prediction positions
    shared_positions = []

    for recording_name, recording_data in dataset.items():
        for bout_name, bout_data in recording_data["bouts"].items():
            whistle_names = sorted(bout_data.keys(),
                                  key=lambda x: int(x.split('_')[1]))
            bout_length = len(whistle_names)

            if bout_length < min_bout_length:
                continue

            labels = [bout_data[w_name]["label"] for w_name in whistle_names]

            for i in range(max_k, bout_length):
                # Only include if target != immediate previous
                if labels[i] != labels[i-1]:
                    shared_positions.append({
                        "recording": recording_name,
                        "bout": bout_name,
                        "position": i,
                        "bout_length": bout_length,
                        "whistle_names": whistle_names
                    })

    # Create test samples for each k
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

            embeddings = [bout_data[w_name]["embedding"] for w_name in whistle_names]
            labels = [bout_data[w_name]["label"] for w_name in whistle_names]

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
    """Balance dataset by undersampling majority classes."""
    label_counts = Counter(y)
    print(f"\n  Original label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"    {label}: {count}")

    min_count = min(label_counts.values())
    print(f"\n  Balancing to {min_count} samples per class...")

    label_to_indices = {label: [] for label in label_counts.keys()}
    for idx, label in enumerate(y):
        label_to_indices[label].append(idx)

    rng = np.random.RandomState(seed)
    balanced_indices = []

    for label, indices in label_to_indices.items():
        if len(indices) >= min_count:
            sampled = rng.choice(indices, size=min_count, replace=False)
        else:
            sampled = indices
        balanced_indices.extend(sampled)

    rng.shuffle(balanced_indices)

    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]
    metadata_balanced = [metadata[i] for i in balanced_indices]

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

    print("="*60)
    print("PREPARING NO-PERSISTENCE DATASET")
    print("="*60)
    print("\nThis version EXCLUDES samples where:")
    print("  target_label == previous_label (no A A patterns)")
    print("\nAllows: A B A, A B C, etc.")
    print("Filters: A A, B B, etc.")

    print("\nLoading dataset...")
    dataset = load_dataset(dataset_path)

    # Split dataset by bouts
    print("\n" + "="*60)
    print("SPLITTING DATASET BY BOUTS")
    print("="*60)
    train_set, val_set, test_set = split_dataset_by_bout(
        dataset, seed=42, train_frac=0.7, val_frac=0.15
    )

    # Context lengths to evaluate
    k_values = [2, 3, 4, 5]

    print("\n" + "="*60)
    print("PREPARING TRAINING DATA (NO PERSISTENCE)")
    print("="*60)

    train_data = {}
    train_data_balanced = {}

    for k in k_values:
        print(f"\nPreparing training data for k={k}...")
        X_train, y_train, metadata_train = create_training_samples_no_persistence(
            train_set, k, min_bout_length=k+1
        )

        print(f"  - Created {len(X_train)} training samples")
        print(f"  - Context shape: {X_train.shape}")
        print(f"  - Target shape: {y_train.shape}")

        train_data[f"k{k}_X"] = X_train
        train_data[f"k{k}_y"] = y_train

        # Balance the data
        print(f"\n  Balancing training data for k={k}...")
        X_train_bal, y_train_bal, metadata_train_bal = balance_by_label(
            X_train, y_train, metadata_train, seed=42
        )

        train_data_balanced[f"k{k}_X"] = X_train_bal
        train_data_balanced[f"k{k}_y"] = y_train_bal

    # Save training data
    save_prepared_data(output_dir + "train_data_no_persist.npz", train_data)
    save_prepared_data(output_dir + "train_data_no_persist_balanced.npz", train_data_balanced)

    print("\n" + "="*60)
    print("PREPARING TEST DATA (NO PERSISTENCE, SHARED POSITIONS)")
    print("="*60)

    test_data, shared_positions = create_test_samples_shared_positions_no_persistence(
        test_set, k_values, min_bout_length=8
    )

    print(f"\nTotal shared prediction positions (no persistence): {len(shared_positions)}")

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

        test_data_dict[f"k{k}_X"] = X_test
        test_data_dict[f"k{k}_y"] = y_test

        # Balance test data
        print(f"\n  Balancing test data for k={k}...")
        X_test_bal, y_test_bal, metadata_test_bal = balance_by_label(
            X_test, y_test, metadata_test, seed=42
        )

        test_data_dict_balanced[f"k{k}_X"] = X_test_bal
        test_data_dict_balanced[f"k{k}_y"] = y_test_bal

    # Save test data
    save_prepared_data(output_dir + "test_data_no_persist.npz", test_data_dict)
    save_prepared_data(output_dir + "test_data_no_persist_balanced.npz", test_data_dict_balanced)

    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"\nFiles saved:")
    print(f"  Unbalanced (no persistence):")
    print(f"    - {output_dir}train_data_no_persist.npz")
    print(f"    - {output_dir}test_data_no_persist.npz")
    print(f"  Balanced (no persistence):")
    print(f"    - {output_dir}train_data_no_persist_balanced.npz")
    print(f"    - {output_dir}test_data_no_persist_balanced.npz")
    print(f"\nNOTE: Train and test sets have NO overlapping bouts!")
    print(f"      All samples exclude label persistence (no A A patterns).")

if __name__ == "__main__":
    main()
