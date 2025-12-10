import json
import numpy as np
from typing import List, Dict, Tuple

def load_dataset(dataset_path):
    """Load the dataset from JSON."""
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    return dataset

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

def save_prepared_data(output_path, data_dict):
    """Save prepared data to disk."""
    np.savez(output_path, **data_dict)
    print(f"Saved data to {output_path}")

def main():
    dataset_path = "../data/dataset.json"
    output_dir = "../data/"

    print("Loading dataset...")
    dataset = load_dataset(dataset_path)

    # Define context lengths to evaluate
    k_values = [2, 3, 4, 5, 6, 7]

    print("\n" + "="*60)
    print("PREPARING TRAINING DATA")
    print("="*60)

    # Prepare training data for each k
    train_data = {}
    for k in k_values:
        print(f"\nPreparing training data for k={k}...")
        X_train, y_train, metadata_train = create_training_samples(
            dataset, k, min_bout_length=k+1
        )
        train_data[f"k{k}_X"] = X_train
        train_data[f"k{k}_y"] = y_train

        print(f"  - Created {len(X_train)} training samples")
        print(f"  - Context shape: {X_train.shape}")
        print(f"  - Target shape: {y_train.shape}")

    # Save training data
    save_prepared_data(output_dir + "train_data.npz", train_data)

    print("\n" + "="*60)
    print("PREPARING TEST DATA (SHARED POSITIONS)")
    print("="*60)

    # Prepare test data with shared positions
    test_data, shared_positions = create_test_samples_shared_positions(
        dataset, k_values, min_bout_length=8
    )

    print(f"\nTotal shared prediction positions: {len(shared_positions)}")

    test_data_dict = {}
    for k in k_values:
        print(f"\nTest data for k={k}:")
        print(f"  - Samples: {len(test_data[k]['X'])}")
        print(f"  - Context shape: {test_data[k]['X'].shape}")
        print(f"  - Target shape: {test_data[k]['y'].shape}")

        test_data_dict[f"k{k}_X"] = test_data[k]["X"]
        test_data_dict[f"k{k}_y"] = test_data[k]["y"]

    # Save test data
    save_prepared_data(output_dir + "test_data.npz", test_data_dict)

    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"\nFiles saved:")
    print(f"  - {output_dir}train_data.npz")
    print(f"  - {output_dir}test_data.npz")

if __name__ == "__main__":
    main()
