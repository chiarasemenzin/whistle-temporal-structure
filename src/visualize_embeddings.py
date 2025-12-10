import json
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP

def extract_embeddings_and_labels(dataset_path):
    """
    Extract all embeddings and labels from the dataset.

    Returns:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        labels: list of labels for each embedding
    """
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    embeddings = []
    labels = []

    for recording_name, recording_data in dataset.items():
        for bout_name, bout_data in recording_data["bouts"].items():
            for whistle_name, whistle_data in bout_data.items():
                embedding = whistle_data.get("embedding")
                label = whistle_data.get("label")

                if embedding is not None and label is not None:
                    embeddings.append(embedding)
                    labels.append(label)

    embeddings = np.array(embeddings)
    return embeddings, labels

def plot_umap(embeddings, labels, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Plot embeddings in 2D using UMAP with colors based on labels.
    """
    # Apply UMAP
    print(f"Applying UMAP to {len(embeddings)} embeddings...")
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    embedding_2d = reducer.fit_transform(embeddings)

    # Get unique labels and assign colors
    unique_labels = sorted(set(labels))
    print(f"Found {len(unique_labels)} unique labels: {unique_labels}")

    # Create color map
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}

    # Plot
    plt.figure(figsize=(12, 8))

    for label in unique_labels:
        mask = np.array([l == label for l in labels])
        plt.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=[label_to_color[label]],
            label=label,
            alpha=0.6,
            s=50
        )

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("Whistle Embeddings Visualization (UMAP)")
    plt.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("../data/embeddings_umap.png", dpi=300, bbox_inches='tight')
    print("Plot saved to ../data/embeddings_umap.png")
    plt.show()

if __name__ == "__main__":
    dataset_path = "../data/dataset.json"

    # Extract embeddings and labels
    embeddings, labels = extract_embeddings_and_labels(dataset_path)
    print(f"Extracted {len(embeddings)} embeddings with shape {embeddings.shape}")

    # Plot using UMAP
    plot_umap(embeddings, labels)
