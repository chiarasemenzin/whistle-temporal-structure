import json
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
import os

def load_embeddings_from_dataset(dataset_file):
    """Load all embeddings from a dataset JSON file."""
    with open(dataset_file, "r") as f:
        dataset = json.load(f)

    embeddings = []
    for recording_name, recording_data in dataset.items():
        for bout_name, bout_data in recording_data["bouts"].items():
            for whistle_name, whistle_data in bout_data.items():
                emb = np.array(whistle_data["embedding"])
                embeddings.append(emb)

    return np.array(embeddings)

def main():
    # Define year folders in the correct order
    year_folders = [
        "2019_2020",
        "start_2021",
        "end_2021",
        "2022",
        "2023",
        "2024_jan-apr"
    ]

    # Create figure with 6 subplots (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    print("Loading embeddings and computing UMAP projections...")

    for idx, year_folder in enumerate(year_folders):
        dataset_file = f"../data/dataset_{year_folder}.json"

        if not os.path.exists(dataset_file):
            print(f"Warning: {dataset_file} not found. Skipping.")
            axes[idx].text(0.5, 0.5, f"Data not found\n{year_folder}",
                          ha='center', va='center', fontsize=12)
            axes[idx].set_title(year_folder)
            continue

        print(f"\nProcessing {year_folder}...")

        # Load embeddings
        embeddings = load_embeddings_from_dataset(dataset_file)
        print(f"  Loaded {len(embeddings)} embeddings")

        # Compute UMAP projection
        print(f"  Computing UMAP...")
        umap_model = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        umap_embeddings = umap_model.fit_transform(embeddings)

        # Plot
        axes[idx].scatter(umap_embeddings[:, 0], umap_embeddings[:, 1],
                         s=0.8, alpha=0.5, c='blue')
        axes[idx].set_title(f"{year_folder}\n({len(embeddings)} whistles)", fontsize=12)
        axes[idx].set_xlabel("UMAP 1")
        axes[idx].set_ylabel("UMAP 2")
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = "umap_embeddings_by_year.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to {output_file}")

    plt.show()

if __name__ == "__main__":
    main()
