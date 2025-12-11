import json
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
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

def reduce_dimensions_pca(embeddings, n_components=10):
    """
    Reduce embeddings to n_components dimensions using PCA.
    This makes convex hull computation more tractable.
    """
    pca = PCA(n_components=n_components)
    embeddings_reduced = pca.fit_transform(embeddings)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    return embeddings_reduced, explained_variance

def compute_convex_hull_volume(embeddings, use_pca=True, n_components=10):
    """
    Compute the volume of the convex hull for a set of embeddings.
    For high-dimensional data, first reduces to n_components dimensions using PCA.
    """
    try:
        if use_pca:
            embeddings_reduced, explained_var = reduce_dimensions_pca(embeddings, n_components)
            print(f"    PCA: reduced to {n_components}D (explained variance: {explained_var:.4f})")
            hull = ConvexHull(embeddings_reduced)
        else:
            hull = ConvexHull(embeddings)
        return hull.volume
    except Exception as e:
        print(f"    Error computing convex hull: {e}")
        return None

def compute_mean_pairwise_distance(embeddings, sample_size=5000):
    """
    Compute the mean pairwise Euclidean distance between embeddings.
    For large datasets, samples a subset to make computation tractable.
    """
    n = len(embeddings)

    # If dataset is large, sample to make computation faster
    if n > sample_size:
        print(f"    Sampling {sample_size} embeddings for pairwise distance computation...")
        indices = np.random.choice(n, sample_size, replace=False)
        embeddings_sample = embeddings[indices]
    else:
        embeddings_sample = embeddings

    # Compute pairwise distances
    distances = pdist(embeddings_sample, metric='euclidean')
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    return mean_distance, std_distance

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

    print("Computing convex hull volumes and mean pairwise distances for each year...\n")
    print("=" * 80)

    results = {}

    for year_folder in year_folders:
        dataset_file = f"../data/dataset_{year_folder}.json"

        if not os.path.exists(dataset_file):
            print(f"{year_folder:20s} | Dataset not found. Skipping.")
            continue

        print(f"Processing {year_folder}...")

        # Load embeddings
        embeddings = load_embeddings_from_dataset(dataset_file)
        print(f"  Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")

        # Compute convex hull volume
        print(f"  Computing convex hull volume...")
        volume = compute_convex_hull_volume(embeddings)

        # Compute mean pairwise distance
        print(f"  Computing mean pairwise distance...")
        mean_dist, std_dist = compute_mean_pairwise_distance(embeddings)

        if volume is not None:
            results[year_folder] = {
                "volume": volume,
                "mean_pairwise_distance": mean_dist,
                "std_pairwise_distance": std_dist,
                "num_embeddings": len(embeddings),
                "dimension": embeddings.shape[1]
            }
            print(f"  Convex hull volume: {volume:.6e}")
            print(f"  Mean pairwise distance: {mean_dist:.6f} ± {std_dist:.6f}")
        else:
            print(f"  Failed to compute convex hull volume")

        print()

    # Summary
    print("=" * 100)
    print("\nSUMMARY")
    print("=" * 100)
    print(f"{'Year':<20s} {'# Whistles':<12s} {'Dimension':<12s} {'Volume':<20s} {'Mean Dist':<15s} {'Std Dist':<15s}")
    print("-" * 100)

    for year_folder in year_folders:
        if year_folder in results:
            r = results[year_folder]
            print(f"{year_folder:<20s} {r['num_embeddings']:<12d} {r['dimension']:<12d} "
                  f"{r['volume']:<20.6e} {r['mean_pairwise_distance']:<15.6f} {r['std_pairwise_distance']:<15.6f}")

    # Save results to JSON
    output_file = "diversity_metrics.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 100)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
