import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score

def load_prepared_data(train_path, test_path):
    """Load prepared training and test data."""
    train_data = np.load(train_path, allow_pickle=True)
    test_data = np.load(test_path, allow_pickle=True)
    return train_data, test_data

def flatten_contexts(X):
    """
    Flatten context embeddings for use with sklearn models.

    Args:
        X: Array of shape (n_samples, k, embedding_dim)

    Returns:
        X_flat: Array of shape (n_samples, k * embedding_dim)
    """
    n_samples, k, embedding_dim = X.shape
    return X.reshape(n_samples, k * embedding_dim)

def train_model(X_train, y_train):
    """
    Train a logistic regression model.

    Args:
        X_train: Training contexts (flattened embeddings)
        y_train: Training target labels (strings)

    Returns:
        model: Trained model
        label_encoder: LabelEncoder for the labels
    """
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Train model
    print(f"  Training logistic regression on {len(X_train)} samples...")
    print(f"  Number of unique labels: {len(label_encoder.classes_)}")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train_encoded)

    return model, label_encoder

def evaluate_model(model, label_encoder, X_test, y_test):
    """
    Evaluate model and return metrics.

    Args:
        model: Trained model
        label_encoder: LabelEncoder for the labels
        X_test: Test contexts (flattened embeddings)
        y_test: Test target labels (strings)

    Returns:
        metrics: Dict with cross_entropy and accuracy
    """
    # Encode test labels
    y_test_encoded = label_encoder.transform(y_test)

    # Get predictions
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    # Calculate metrics
    cross_entropy = log_loss(y_test_encoded, y_pred_proba)
    accuracy = accuracy_score(y_test_encoded, y_pred)

    return {
        "cross_entropy": cross_entropy,
        "accuracy": accuracy
    }

def main():
    # Paths
    train_path = "../data/train_data.npz"
    test_path = "../data/test_data.npz"

    print("Loading prepared data...")
    train_data, test_data = load_prepared_data(train_path, test_path)

    # Context lengths to evaluate
    k_values = [2, 3, 4, 5, 6, 7]

    results = {
        "k_values": k_values,
        "cross_entropy": [],
        "accuracy": []
    }

    print("\n" + "="*60)
    print("TRAINING AND EVALUATION")
    print("="*60)

    for k in k_values:
        print(f"\n{'='*60}")
        print(f"Context length k = {k}")
        print(f"{'='*60}")

        # Load training data for this k
        X_train = train_data[f"k{k}_X"]
        y_train = train_data[f"k{k}_y"]  # labels now, not embeddings

        # Load test data for this k
        X_test = test_data[f"k{k}_X"]
        y_test = test_data[f"k{k}_y"]  # labels now, not embeddings

        # Flatten contexts (concatenate embeddings)
        X_train_flat = flatten_contexts(X_train)
        X_test_flat = flatten_contexts(X_test)

        print(f"  Training samples: {len(X_train_flat)}")
        print(f"  Test samples: {len(X_test_flat)}")

        # Train model
        model, label_encoder = train_model(X_train_flat, y_train)

        # Evaluate model
        print("  Evaluating model...")
        metrics = evaluate_model(model, label_encoder, X_test_flat, y_test)

        print(f"  Cross-entropy: {metrics['cross_entropy']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")

        # Store results
        results["cross_entropy"].append(metrics["cross_entropy"])
        results["accuracy"].append(metrics["accuracy"])

    # Plot results
    print("\n" + "="*60)
    print("PLOTTING RESULTS")
    print("="*60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot cross-entropy
    ax1.plot(results["k_values"], results["cross_entropy"], 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel("Context Length (k)", fontsize=12)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax1.set_title("Cross-Entropy vs Context Length", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(results["k_values"])

    # Plot accuracy
    ax2.plot(results["k_values"], results["accuracy"], 'o-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel("Context Length (k)", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Accuracy vs Context Length", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(results["k_values"])

    plt.tight_layout()
    plt.savefig("../data/context_length_evaluation.png", dpi=300, bbox_inches='tight')
    print("Plot saved to ../data/context_length_evaluation.png")
    plt.show()

    # Save results
    np.savez("../data/evaluation_results.npz", **results)
    print("Results saved to ../data/evaluation_results.npz")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nContext Length | Cross-Entropy | Accuracy")
    print("-" * 45)
    for i, k in enumerate(results["k_values"]):
        ce = results["cross_entropy"][i]
        acc = results["accuracy"][i]
        print(f"      {k}        |    {ce:.4f}     | {acc:.4f}")

    # Analyze trend
    ce_diff = results["cross_entropy"][-1] - results["cross_entropy"][0]
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    if abs(ce_diff) < 0.1:
        print("Cross-entropy is relatively flat → weak temporal dependence")
    elif ce_diff < -0.1:
        print("Cross-entropy decreases with context → increasing context helps!")
    else:
        print("Cross-entropy increases with context → longer context may hurt")

if __name__ == "__main__":
    main()
