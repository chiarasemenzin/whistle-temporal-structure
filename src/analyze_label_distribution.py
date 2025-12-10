import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils import resample

def analyze_label_distribution(data_path, dataset_name):
    """
    Analyze and visualize label distribution in a dataset.

    Args:
        data_path: Path to the .npz file
        dataset_name: Name of the dataset (for printing)

    Returns:
        distributions: Dict mapping k -> label counts
    """
    print(f"\n{'='*60}")
    print(f"{dataset_name.upper()} LABEL DISTRIBUTION")
    print(f"{'='*60}")

    data = np.load(data_path, allow_pickle=True)
    k_values = [2, 3, 4, 5, 6, 7]
    distributions = {}

    for k in k_values:
        y_key = f"k{k}_y"
        if y_key in data:
            y = data[y_key]
            label_counts = Counter(y)
            distributions[k] = label_counts

            print(f"\nContext length k={k}:")
            print(f"  Total samples: {len(y)}")
            print(f"  Number of unique labels: {len(label_counts)}")
            print(f"\n  Label distribution:")

            # Sort by count (descending)
            for label, count in label_counts.most_common():
                percentage = (count / len(y)) * 100
                print(f"    {label}: {count:4d} ({percentage:5.2f}%)")

            # Calculate imbalance ratio
            max_count = max(label_counts.values())
            min_count = min(label_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            print(f"\n  Imbalance ratio (max/min): {imbalance_ratio:.2f}")

    return distributions

def visualize_distributions(train_dists, test_dists, output_path="../data/label_distributions.png"):
    """Create visualization of label distributions."""
    k_values = sorted(train_dists.keys())

    fig, axes = plt.subplots(2, len(k_values), figsize=(18, 8))

    for idx, k in enumerate(k_values):
        # Training distribution
        train_labels = list(train_dists[k].keys())
        train_counts = list(train_dists[k].values())

        axes[0, idx].bar(range(len(train_labels)), train_counts)
        axes[0, idx].set_title(f'Train k={k}', fontsize=10)
        axes[0, idx].set_xlabel('Label', fontsize=8)
        axes[0, idx].set_ylabel('Count', fontsize=8)
        axes[0, idx].set_xticks(range(len(train_labels)))
        axes[0, idx].set_xticklabels(train_labels, rotation=45, ha='right', fontsize=7)
        axes[0, idx].grid(True, alpha=0.3)

        # Test distribution
        test_labels = list(test_dists[k].keys())
        test_counts = list(test_dists[k].values())

        axes[1, idx].bar(range(len(test_labels)), test_counts, color='orange')
        axes[1, idx].set_title(f'Test k={k}', fontsize=10)
        axes[1, idx].set_xlabel('Label', fontsize=8)
        axes[1, idx].set_ylabel('Count', fontsize=8)
        axes[1, idx].set_xticks(range(len(test_labels)))
        axes[1, idx].set_xticklabels(test_labels, rotation=45, ha='right', fontsize=7)
        axes[1, idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {output_path}")
    plt.close()

def balance_dataset_by_undersampling(X, y, random_state=42):
    """
    Balance dataset by undersampling majority classes.

    Args:
        X: Feature array
        y: Label array
        random_state: Random seed for reproducibility

    Returns:
        X_balanced: Balanced feature array
        y_balanced: Balanced label array
    """
    label_counts = Counter(y)
    min_count = min(label_counts.values())

    print(f"\n  Undersampling to {min_count} samples per class...")

    X_balanced_list = []
    y_balanced_list = []

    for label in label_counts.keys():
        # Get indices for this label
        label_indices = np.where(y == label)[0]

        # Undersample to min_count
        if len(label_indices) > min_count:
            label_indices = resample(
                label_indices,
                n_samples=min_count,
                replace=False,
                random_state=random_state
            )

        X_balanced_list.append(X[label_indices])
        y_balanced_list.append(y[label_indices])

    X_balanced = np.concatenate(X_balanced_list, axis=0)
    y_balanced = np.concatenate(y_balanced_list, axis=0)

    # Shuffle the balanced dataset
    shuffle_indices = np.random.RandomState(random_state).permutation(len(X_balanced))
    X_balanced = X_balanced[shuffle_indices]
    y_balanced = y_balanced[shuffle_indices]

    return X_balanced, y_balanced

def balance_dataset_by_oversampling(X, y, random_state=42):
    """
    Balance dataset by oversampling minority classes.

    Args:
        X: Feature array
        y: Label array
        random_state: Random seed for reproducibility

    Returns:
        X_balanced: Balanced feature array
        y_balanced: Balanced label array
    """
    label_counts = Counter(y)
    max_count = max(label_counts.values())

    print(f"\n  Oversampling to {max_count} samples per class...")

    X_balanced_list = []
    y_balanced_list = []

    for label in label_counts.keys():
        # Get indices for this label
        label_indices = np.where(y == label)[0]

        # Oversample to max_count
        if len(label_indices) < max_count:
            label_indices = resample(
                label_indices,
                n_samples=max_count,
                replace=True,  # Allow duplicates
                random_state=random_state
            )

        X_balanced_list.append(X[label_indices])
        y_balanced_list.append(y[label_indices])

    X_balanced = np.concatenate(X_balanced_list, axis=0)
    y_balanced = np.concatenate(y_balanced_list, axis=0)

    # Shuffle the balanced dataset
    shuffle_indices = np.random.RandomState(random_state).permutation(len(X_balanced))
    X_balanced = X_balanced[shuffle_indices]
    y_balanced = y_balanced[shuffle_indices]

    return X_balanced, y_balanced

def create_balanced_datasets(train_path, test_path, method='undersample', random_state=42):
    """
    Create balanced versions of train and test datasets.

    Args:
        train_path: Path to training data
        test_path: Path to test data
        method: 'undersample' or 'oversample'
        random_state: Random seed

    Returns:
        balanced_train_data: Dict with balanced training data
        balanced_test_data: Dict with balanced test data
    """
    print(f"\n{'='*60}")
    print(f"CREATING BALANCED DATASETS (method: {method})")
    print(f"{'='*60}")

    train_data = np.load(train_path, allow_pickle=True)
    test_data = np.load(test_path, allow_pickle=True)

    k_values = [2, 3, 4, 5, 6, 7]

    balanced_train_data = {}
    balanced_test_data = {}

    balance_func = balance_dataset_by_undersampling if method == 'undersample' else balance_dataset_by_oversampling

    for k in k_values:
        print(f"\nBalancing k={k}...")

        # Balance training data
        X_train = train_data[f"k{k}_X"]
        y_train = train_data[f"k{k}_y"]

        print(f"  Training data:")
        print(f"    Original: {len(X_train)} samples")
        X_train_balanced, y_train_balanced = balance_func(X_train, y_train, random_state)
        print(f"    Balanced: {len(X_train_balanced)} samples")

        balanced_train_data[f"k{k}_X"] = X_train_balanced
        balanced_train_data[f"k{k}_y"] = y_train_balanced

        # Balance test data
        X_test = test_data[f"k{k}_X"]
        y_test = test_data[f"k{k}_y"]

        print(f"  Test data:")
        print(f"    Original: {len(X_test)} samples")
        X_test_balanced, y_test_balanced = balance_func(X_test, y_test, random_state)
        print(f"    Balanced: {len(X_test_balanced)} samples")

        balanced_test_data[f"k{k}_X"] = X_test_balanced
        balanced_test_data[f"k{k}_y"] = y_test_balanced

    return balanced_train_data, balanced_test_data

def main():
    train_path = "../data/train_data.npz"
    test_path = "../data/test_data.npz"

    # Analyze original distributions
    print("\n" + "="*60)
    print("ANALYZING ORIGINAL DISTRIBUTIONS")
    print("="*60)

    train_dists = analyze_label_distribution(train_path, "Training")
    test_dists = analyze_label_distribution(test_path, "Test")

    # Visualize distributions
    visualize_distributions(train_dists, test_dists)

    # Create balanced datasets
    print("\n" + "="*60)
    print("BALANCING OPTIONS")
    print("="*60)
    print("\nTwo methods are available:")
    print("  1. Undersampling: Reduce majority classes to match minority class")
    print("  2. Oversampling: Increase minority classes to match majority class (with repetition)")
    print("\nYou can modify the 'method' parameter in the code to choose.")

    # Example: Create balanced datasets using undersampling
    balanced_train_data, balanced_test_data = create_balanced_datasets(
        train_path, test_path, method='undersample', random_state=42
    )

    # Save balanced datasets
    np.savez("../data/train_data_balanced.npz", **balanced_train_data)
    np.savez("../data/test_data_balanced.npz", **balanced_test_data)

    print("\n" + "="*60)
    print("BALANCED DATASETS SAVED")
    print("="*60)
    print("  - ../data/train_data_balanced.npz")
    print("  - ../data/test_data_balanced.npz")

    # Analyze balanced distributions
    print("\n" + "="*60)
    print("ANALYZING BALANCED DISTRIBUTIONS")
    print("="*60)

    train_dists_balanced = analyze_label_distribution("../data/train_data_balanced.npz", "Training (Balanced)")
    test_dists_balanced = analyze_label_distribution("../data/test_data_balanced.npz", "Test (Balanced)")

    # Visualize balanced distributions
    visualize_distributions(
        train_dists_balanced,
        test_dists_balanced,
        output_path="../data/label_distributions_balanced.png"
    )

if __name__ == "__main__":
    main()
