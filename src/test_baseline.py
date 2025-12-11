import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
from collections import Counter

def load_dataset(dataset_path):
    """Load the dataset from JSON."""
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    return dataset

def test_persistence_baseline(dataset):
    """
    Test a baseline that always predicts "same label as previous".

    For each whistle, predict its label to be the same as the previous whistle.

    Args:
        dataset: The dataset dictionary

    Returns:
        metrics: Dict with accuracy and cross-entropy
    """
    y_true = []  # actual labels
    y_pred = []  # predicted labels (previous label)

    for recording_name, recording_data in dataset.items():
        for bout_name, bout_data in recording_data["bouts"].items():
            # Get all whistles in this bout
            whistle_names = sorted(bout_data.keys(),
                                  key=lambda x: int(x.split('_')[1]))

            if len(whistle_names) < 2:
                continue  # Need at least 2 whistles

            # Extract labels for this bout
            labels = [bout_data[w_name]["label"] for w_name in whistle_names]

            # For each whistle after the first, predict previous label
            for i in range(1, len(labels)):
                y_true.append(labels[i])
                y_pred.append(labels[i-1])  # Predict same as previous

    return y_true, y_pred

def test_majority_baseline(y_true, y_pred_persistence):
    """
    Test a baseline that always predicts the most common label.

    Args:
        y_true: True labels
        y_pred_persistence: Predictions from persistence baseline (for comparison)

    Returns:
        y_pred_majority: Predictions using majority class
    """
    # Find most common label
    label_counts = Counter(y_true)
    majority_label = label_counts.most_common(1)[0][0]

    print(f"\nMajority class: {majority_label} ({label_counts[majority_label]} / {len(y_true)} = {label_counts[majority_label]/len(y_true)*100:.1f}%)")

    # Predict majority for everything
    y_pred_majority = [majority_label] * len(y_true)

    return y_pred_majority

def calculate_metrics(y_true, y_pred, baseline_name):
    """Calculate and print metrics for a baseline."""
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(y_true)

    y_true_encoded = label_encoder.transform(y_true)
    y_pred_encoded = label_encoder.transform(y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_true_encoded, y_pred_encoded)

    # Calculate cross-entropy
    # For deterministic predictions, create a one-hot probability distribution
    n_classes = len(label_encoder.classes_)
    y_pred_proba = np.zeros((len(y_pred_encoded), n_classes))
    y_pred_proba[np.arange(len(y_pred_encoded)), y_pred_encoded] = 1.0

    cross_entropy = log_loss(y_true_encoded, y_pred_proba)

    # Print results
    print(f"\n{baseline_name} Results:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Cross-Entropy: {cross_entropy:.4f}")

    # Label distribution
    print(f"\n  True label distribution:")
    true_counts = Counter(y_true)
    for label, count in sorted(true_counts.items()):
        print(f"    {label}: {count} ({count/len(y_true)*100:.1f}%)")

    print(f"\n  Predicted label distribution:")
    pred_counts = Counter(y_pred)
    for label, count in sorted(pred_counts.items()):
        print(f"    {label}: {count} ({count/len(y_pred)*100:.1f}%)")

    return {
        "accuracy": accuracy,
        "cross_entropy": cross_entropy
    }

def compare_with_model_results(baseline_metrics, model_results_path="../data/evaluation_results.npz"):
    """
    Compare baseline results with trained model results.

    Args:
        baseline_metrics: Dict with baseline accuracy and cross_entropy
        model_results_path: Path to saved model evaluation results
    """
    try:
        results = np.load(model_results_path, allow_pickle=True)
        k_values = results["k_values"]
        model_ce = results["cross_entropy"]
        model_acc = results["accuracy"]

        print("\n" + "="*60)
        print("COMPARISON WITH TRAINED MODELS")
        print("="*60)

        print("\nContext Length | Model Acc | Baseline Acc | Model CE | Baseline CE")
        print("-" * 70)
        for i, k in enumerate(k_values):
            print(f"      {k}        | {model_acc[i]:8.4f}  | {baseline_metrics['accuracy']:12.4f} | {model_ce[i]:8.4f} | {baseline_metrics['cross_entropy']:11.4f}")

        print("\n" + "="*60)
        print("INTERPRETATION")
        print("="*60)

        best_model_acc = max(model_acc)
        best_model_ce = min(model_ce)

        if best_model_acc > baseline_metrics['accuracy'] + 0.01:
            print(f"✓ Model outperforms persistence baseline on accuracy")
            print(f"  Best model: {best_model_acc:.4f} vs Baseline: {baseline_metrics['accuracy']:.4f}")
        else:
            print(f"✗ Model does NOT significantly outperform baseline on accuracy")
            print(f"  Best model: {best_model_acc:.4f} vs Baseline: {baseline_metrics['accuracy']:.4f}")

        if best_model_ce < baseline_metrics['cross_entropy'] - 0.01:
            print(f"✓ Model outperforms persistence baseline on cross-entropy")
            print(f"  Best model: {best_model_ce:.4f} vs Baseline: {baseline_metrics['cross_entropy']:.4f}")
        else:
            print(f"✗ Model does NOT significantly outperform baseline on cross-entropy")
            print(f"  Best model: {best_model_ce:.4f} vs Baseline: {baseline_metrics['cross_entropy']:.4f}")

    except FileNotFoundError:
        print(f"\nModel results not found at {model_results_path}")
        print("Run train_and_evaluate.py first to generate model results.")

def main():
    dataset_path = "../data/dataset.json"

    print("="*60)
    print("BASELINE EVALUATION: PERSISTENCE PREDICTOR")
    print("="*60)
    print("\nThis baseline predicts each whistle's label to be")
    print("the same as the previous whistle in the bout.")

    print("\nLoading dataset...")
    dataset = load_dataset(dataset_path)

    print("\nTesting persistence baseline...")
    y_true, y_pred_persistence = test_persistence_baseline(dataset)

    print(f"\nTotal predictions: {len(y_true)}")

    # Calculate persistence baseline metrics
    persistence_metrics = calculate_metrics(y_true, y_pred_persistence, "PERSISTENCE BASELINE")

    print("\n" + "="*60)
    print("BASELINE EVALUATION: MAJORITY CLASS PREDICTOR")
    print("="*60)
    print("\nThis baseline always predicts the most common label.")

    y_pred_majority = test_majority_baseline(y_true, y_pred_persistence)

    # Calculate majority baseline metrics
    majority_metrics = calculate_metrics(y_true, y_pred_majority, "MAJORITY CLASS BASELINE")

    print("\n" + "="*60)
    print("BASELINE COMPARISON")
    print("="*60)
    print(f"\nPersistence Baseline:")
    print(f"  Accuracy: {persistence_metrics['accuracy']:.4f}")
    print(f"  Cross-Entropy: {persistence_metrics['cross_entropy']:.4f}")

    print(f"\nMajority Class Baseline:")
    print(f"  Accuracy: {majority_metrics['accuracy']:.4f}")
    print(f"  Cross-Entropy: {majority_metrics['cross_entropy']:.4f}")

    if persistence_metrics['accuracy'] > majority_metrics['accuracy']:
        print("\n→ Persistence baseline is better than majority class!")
        print("  This suggests label persistence is a strong signal.")
    else:
        print("\n→ Majority class baseline is better than persistence.")
        print("  This suggests labels change frequently within bouts.")

    # Compare with model results if available
    compare_with_model_results(persistence_metrics)

if __name__ == "__main__":
    main()
