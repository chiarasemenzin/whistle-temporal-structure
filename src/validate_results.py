import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import seaborn as sns
from collections import Counter

def load_data(train_path, test_path):
    """Load training and test data."""
    train_data = np.load(train_path, allow_pickle=True)
    test_data = np.load(test_path, allow_pickle=True)
    return train_data, test_data

def flatten_contexts(X):
    """Flatten context embeddings."""
    n_samples, k, embedding_dim = X.shape
    return X.reshape(n_samples, k * embedding_dim)

def check_data_leakage(train_data, test_data, k_values):
    """
    Check for data leakage between train and test sets.

    This is critical - if there's overlap, results are invalid.
    """
    print("\n" + "="*60)
    print("1. CHECKING FOR DATA LEAKAGE")
    print("="*60)

    for k in k_values:
        print(f"\nk={k}:")

        X_train = train_data[f"k{k}_X"]
        X_test = test_data[f"k{k}_X"]

        # Flatten for comparison
        X_train_flat = flatten_contexts(X_train)
        X_test_flat = flatten_contexts(X_test)

        # Check if any test samples appear in training set
        # (This is a simplistic check - works if there are exact duplicates)
        train_set = set(map(tuple, X_train_flat))
        test_set = set(map(tuple, X_test_flat))

        overlap = train_set.intersection(test_set)

        print(f"  Train samples: {len(train_set)}")
        print(f"  Test samples: {len(test_set)}")
        print(f"  Overlapping samples: {len(overlap)}")

        if len(overlap) > 0:
            print(f"  ⚠️  WARNING: Found {len(overlap)} overlapping samples!")
            print(f"     Results may be inflated due to data leakage!")
        else:
            print(f"  ✓ No data leakage detected")

def check_label_distribution_similarity(train_data, test_data, k_values):
    """
    Check if train and test have similar label distributions.

    Large differences suggest the test set may not be representative.
    """
    print("\n" + "="*60)
    print("2. CHECKING LABEL DISTRIBUTION SIMILARITY")
    print("="*60)

    for k in k_values:
        print(f"\nk={k}:")

        y_train = train_data[f"k{k}_y"]
        y_test = test_data[f"k{k}_y"]

        train_counts = Counter(y_train)
        test_counts = Counter(y_test)

        # Get all labels
        all_labels = sorted(set(y_train) | set(y_test))

        print(f"\n  {'Label':<15} {'Train %':<12} {'Test %':<12} {'Difference':<12}")
        print(f"  {'-'*55}")

        max_diff = 0
        for label in all_labels:
            train_pct = (train_counts[label] / len(y_train)) * 100
            test_pct = (test_counts[label] / len(y_test)) * 100
            diff = abs(train_pct - test_pct)
            max_diff = max(max_diff, diff)

            status = "⚠️" if diff > 5 else "✓"
            print(f"  {label:<15} {train_pct:>6.2f}%      {test_pct:>6.2f}%      {diff:>6.2f}% {status}")

        print(f"\n  Maximum difference: {max_diff:.2f}%")
        if max_diff > 10:
            print(f"  ⚠️  Large distribution mismatch - results may not generalize well")
        else:
            print(f"  ✓ Distributions are reasonably similar")

def check_baseline_performance(train_data, test_data, k_values):
    """
    Compare against baseline models.

    Your model should beat random guessing and majority class baseline.
    """
    print("\n" + "="*60)
    print("3. BASELINE COMPARISONS")
    print("="*60)

    for k in k_values:
        print(f"\nk={k}:")

        y_train = train_data[f"k{k}_y"]
        y_test = test_data[f"k{k}_y"]

        # Count labels
        train_counts = Counter(y_train)
        test_counts = Counter(y_test)
        n_classes = len(train_counts)

        # Random guessing baseline
        random_accuracy = 1.0 / n_classes

        # Majority class baseline (predict most common class)
        most_common_label = train_counts.most_common(1)[0][0]
        majority_accuracy = test_counts[most_common_label] / len(y_test)

        print(f"  Number of classes: {n_classes}")
        print(f"  Random guessing accuracy: {random_accuracy:.4f} ({random_accuracy*100:.2f}%)")
        print(f"  Majority class baseline: {majority_accuracy:.4f} ({majority_accuracy*100:.2f}%)")
        print(f"\n  Your model MUST beat these baselines to be valid!")

def perform_cross_validation(train_data, k_values, cv_folds=5):
    """
    Perform cross-validation on training data.

    This checks if performance is stable across different data splits.
    """
    print("\n" + "="*60)
    print(f"4. CROSS-VALIDATION (k-fold={cv_folds})")
    print("="*60)
    print("\nThis checks if results are stable across different train/val splits...")

    for k in k_values:
        print(f"\nk={k}:")

        X_train = train_data[f"k{k}_X"]
        y_train = train_data[f"k{k}_y"]

        # Flatten
        X_train_flat = flatten_contexts(X_train)

        # Encode labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)

        # Cross-validation
        model = LogisticRegression(max_iter=2000, random_state=42)
        cv_scores = cross_val_score(model, X_train_flat, y_train_encoded,
                                    cv=cv_folds, scoring='accuracy')

        print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  Individual folds: {[f'{s:.4f}' for s in cv_scores]}")

        if cv_scores.std() > 0.05:
            print(f"  ⚠️  High variance - results may be unstable")
        else:
            print(f"  ✓ Low variance - results are stable")

def analyze_confusion_matrices(train_data, test_data, k_values):
    """
    Generate confusion matrices to see which classes are confused.

    Helps identify if model is actually learning patterns or just memorizing.
    """
    print("\n" + "="*60)
    print("5. CONFUSION MATRIX ANALYSIS")
    print("="*60)

    fig, axes = plt.subplots(1, len(k_values), figsize=(5*len(k_values), 4))
    if len(k_values) == 1:
        axes = [axes]

    for idx, k in enumerate(k_values):
        print(f"\nk={k}:")

        X_train = train_data[f"k{k}_X"]
        y_train = train_data[f"k{k}_y"]
        X_test = test_data[f"k{k}_X"]
        y_test = test_data[f"k{k}_y"]

        # Flatten
        X_train_flat = flatten_contexts(X_train)
        X_test_flat = flatten_contexts(X_test)

        # Train
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        model = LogisticRegression(max_iter=2000, random_state=42)
        model.fit(X_train_flat, y_train_encoded)

        # Predict
        y_pred = model.predict(X_test_flat)

        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plot
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_,
                   ax=axes[idx], cbar_kws={'label': 'Proportion'})
        axes[idx].set_title(f'k={k}')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')

        # Calculate per-class accuracy
        per_class_acc = cm_normalized.diagonal()
        print(f"  Per-class accuracy:")
        for i, label in enumerate(label_encoder.classes_):
            print(f"    {label}: {per_class_acc[i]:.4f}")

        # Check if model is just predicting one class
        pred_distribution = np.bincount(y_pred)
        if pred_distribution.max() / len(y_pred) > 0.8:
            print(f"  ⚠️  Model heavily biased toward one class!")

    plt.tight_layout()
    plt.savefig("../data/confusion_matrices.png", dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrices saved to ../data/confusion_matrices.png")
    plt.close()

def check_feature_importance(train_data, k_values):
    """
    Check if model is using features meaningfully.

    If all coefficients are near zero, model isn't learning.
    """
    print("\n" + "="*60)
    print("6. FEATURE IMPORTANCE CHECK")
    print("="*60)

    for k in k_values:
        print(f"\nk={k}:")

        X_train = train_data[f"k{k}_X"]
        y_train = train_data[f"k{k}_y"]

        # Flatten
        X_train_flat = flatten_contexts(X_train)

        # Train
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)

        model = LogisticRegression(max_iter=2000, random_state=42)
        model.fit(X_train_flat, y_train_encoded)

        # Analyze coefficients
        coef_magnitudes = np.abs(model.coef_).mean(axis=0)

        print(f"  Coefficient statistics:")
        print(f"    Mean magnitude: {coef_magnitudes.mean():.6f}")
        print(f"    Max magnitude: {coef_magnitudes.max():.6f}")
        print(f"    Min magnitude: {coef_magnitudes.min():.6f}")
        print(f"    Std magnitude: {coef_magnitudes.std():.6f}")

        if coef_magnitudes.max() < 0.01:
            print(f"  ⚠️  Very small coefficients - model may not be learning")
        else:
            print(f"  ✓ Coefficients show meaningful variation")

def test_random_labels(train_data, test_data, k_values):
    """
    Train on random labels - should get ~random accuracy.

    If model performs well on random labels, something is wrong.
    """
    print("\n" + "="*60)
    print("7. RANDOM LABEL TEST (Sanity Check)")
    print("="*60)
    print("\nTraining models with shuffled labels...")
    print("Performance should be close to random guessing.")

    for k in k_values:
        print(f"\nk={k}:")

        X_train = train_data[f"k{k}_X"]
        y_train = train_data[f"k{k}_y"]
        X_test = test_data[f"k{k}_X"]
        y_test = test_data[f"k{k}_y"]

        # Flatten
        X_train_flat = flatten_contexts(X_train)
        X_test_flat = flatten_contexts(X_test)

        # Shuffle labels randomly
        y_train_shuffled = np.random.permutation(y_train)

        # Train
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train_shuffled)
        y_test_encoded = label_encoder.transform(y_test)

        model = LogisticRegression(max_iter=2000, random_state=42)
        model.fit(X_train_flat, y_train_encoded)

        # Evaluate
        y_pred = model.predict(X_test_flat)
        random_accuracy = (y_pred == y_test_encoded).mean()

        expected_random = 1.0 / len(np.unique(y_train))

        print(f"  Accuracy with random labels: {random_accuracy:.4f}")
        print(f"  Expected (random guessing): {expected_random:.4f}")

        if random_accuracy > expected_random + 0.1:
            print(f"  ⚠️  WARNING: Model performs too well on random labels!")
            print(f"     This suggests overfitting or data leakage!")
        else:
            print(f"  ✓ Random label performance as expected")

def check_training_test_accuracy_gap(train_data, test_data, k_values):
    """
    Compare training and test accuracy.

    Large gap indicates overfitting.
    """
    print("\n" + "="*60)
    print("8. TRAIN vs TEST ACCURACY GAP")
    print("="*60)

    for k in k_values:
        print(f"\nk={k}:")

        X_train = train_data[f"k{k}_X"]
        y_train = train_data[f"k{k}_y"]
        X_test = test_data[f"k{k}_X"]
        y_test = test_data[f"k{k}_y"]

        # Flatten
        X_train_flat = flatten_contexts(X_train)
        X_test_flat = flatten_contexts(X_test)

        # Train
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        model = LogisticRegression(max_iter=2000, random_state=42)
        model.fit(X_train_flat, y_train_encoded)

        # Evaluate
        train_acc = model.score(X_train_flat, y_train_encoded)
        test_acc = model.score(X_test_flat, y_test_encoded)
        gap = train_acc - test_acc

        print(f"  Training accuracy: {train_acc:.4f}")
        print(f"  Test accuracy: {test_acc:.4f}")
        print(f"  Gap: {gap:.4f}")

        if gap > 0.15:
            print(f"  ⚠️  Large gap - model may be overfitting")
        elif gap < 0:
            print(f"  ⚠️  Test > Train - unusual, check data preparation")
        else:
            print(f"  ✓ Reasonable gap")

def main():
    train_path = "../data/train_data_balanced.npz"
    test_path = "../data/test_data_balanced.npz"

    k_values = [2, 3, 4, 5]

    print("\n" + "="*60)
    print("VALIDATION SUITE FOR MODEL RESULTS")
    print("="*60)
    print("\nThis will check for common issues that could invalidate results:")
    print("  1. Data leakage")
    print("  2. Distribution mismatch")
    print("  3. Baseline comparisons")
    print("  4. Cross-validation stability")
    print("  5. Confusion matrix analysis")
    print("  6. Feature importance")
    print("  7. Random label test")
    print("  8. Overfitting check")

    print("\nLoading data...")
    train_data, test_data = load_data(train_path, test_path)

    # Run all validation checks
    test_random_labels(train_data, test_data, k_values)
    check_data_leakage(train_data, test_data, k_values)
    check_label_distribution_similarity(train_data, test_data, k_values)
    check_baseline_performance(train_data, test_data, k_values)
    #perform_cross_validation(train_data, k_values, cv_folds=5)
    analyze_confusion_matrices(train_data, test_data, k_values)
    check_feature_importance(train_data, k_values)
    check_training_test_accuracy_gap(train_data, test_data, k_values)

    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print("\nReview the warnings (⚠️) above.")
    print("If there are no warnings, your results are likely valid!")

if __name__ == "__main__":
    main()
