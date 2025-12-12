import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import hdbscan
from scipy.stats import entropy
import re

def load_dataset(filepath):
    """Load a single dataset JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)

def extract_year_from_recording(recording_name):
    """
    Extract year from recording name.

    Recording formats:
    - Format 1: Exp_DD_Mon_YYYY_HHMMam/pm (2019-2020 data)
    - Format 2: Exp_DD_Mon_YYYY_HHMM_channel_N (2022-2024 data)

    Returns:
        year: Integer year (e.g., 2019, 2022)
    """
    # Try pattern 1: Exp_DD_Mon_YYYY_HHMM_channel_N (newer format)
    pattern1 = r'Exp_(\d{1,2})_([A-Za-z]+)_(\d{4})_\d{4}_channel_\d+'
    match = re.search(pattern1, recording_name)

    if not match:
        # Try pattern 2: Exp_DD_Mon_YYYY_HHMMam/pm (older format)
        pattern2 = r'Exp_(\d{1,2})_([A-Za-z]+)_(\d{4})_\d{4}(am|pm)'
        match = re.search(pattern2, recording_name)

    if match:
        year = int(match.group(3))
        return year

    return None

def collect_embeddings_with_metadata(data_dir, dataset_files):
    """
    Collect all embeddings along with metadata from multiple datasets.

    Returns:
        embeddings: Array of embeddings
        years: List of years
        recording_names: List of recording names
        labels: List of whistle labels
    """
    embeddings = []
    years = []
    recording_names = []
    labels = []

    skipped_recordings = 0

    for filename in dataset_files:
        filepath = f"{data_dir}/{filename}"
        print(f"Loading {filename}...")

        try:
            dataset = load_dataset(filepath)
            print(f"  Found {len(dataset)} recordings")
        except FileNotFoundError:
            print(f"  Warning: {filepath} not found, skipping")
            continue

        # Process each recording
        for recording_name, recording_data in dataset.items():
            # Extract year from recording name
            year = extract_year_from_recording(recording_name)

            if year is None:
                skipped_recordings += 1
                continue

            # Extract embeddings from all bouts
            for bout_name, bout_data in recording_data["bouts"].items():
                for whistle_name, whistle_data in bout_data.items():
                    embedding = whistle_data.get("embedding")
                    label = whistle_data.get("label")

                    if embedding is not None:
                        embeddings.append(np.array(embedding))
                        years.append(year)
                        recording_names.append(recording_name)
                        labels.append(label)

    if skipped_recordings > 0:
        print(f"\nWarning: Skipped {skipped_recordings} recordings (couldn't extract year)")

    return np.array(embeddings), years, recording_names, labels

def compute_cluster_statistics(cluster_labels, years):
    """
    Compute statistics for each cluster.

    Args:
        cluster_labels: Array of cluster assignments for each embedding
        years: List of years for each embedding

    Returns:
        cluster_stats: Dictionary with statistics for each cluster
    """
    unique_clusters = np.unique(cluster_labels)
    unique_years = sorted(set(years))

    cluster_stats = {}

    for cluster_id in unique_clusters:
        if cluster_id == -1:
            cluster_name = "Noise"
        else:
            cluster_name = f"Cluster {cluster_id}"

        # Get indices for this cluster
        mask = cluster_labels == cluster_id
        cluster_years = [years[i] for i in range(len(years)) if mask[i]]

        # Count per year: N_k(y)
        count_per_year = {}
        for year in unique_years:
            count_per_year[year] = cluster_years.count(year)

        # Total whistles in cluster: |W_k|
        total_whistles = len(cluster_years)

        # Year distribution: p_k(y) = N_k(y) / |W_k|
        year_distribution = {}
        for year in unique_years:
            year_distribution[year] = count_per_year[year] / total_whistles if total_whistles > 0 else 0

        # Entropy of year distribution
        # H(p_k) = -sum_y p_k(y) * log(p_k(y))
        probabilities = [year_distribution[year] for year in unique_years]
        year_entropy = entropy(probabilities, base=2)  # Use base 2 for bits

        cluster_stats[cluster_id] = {
            "name": cluster_name,
            "total_whistles": total_whistles,
            "count_per_year": count_per_year,
            "year_distribution": year_distribution,
            "year_entropy": year_entropy
        }

    return cluster_stats, unique_years

def print_cluster_statistics(cluster_stats, unique_years):
    """Print formatted statistics for each cluster."""
    print("\n" + "="*70)
    print("CLUSTER STATISTICS")
    print("="*70)

    # Sort clusters by ID (noise at the end)
    cluster_ids = sorted([cid for cid in cluster_stats.keys() if cid != -1])
    if -1 in cluster_stats:
        cluster_ids.append(-1)

    for cluster_id in cluster_ids:
        stats = cluster_stats[cluster_id]

        print(f"\n{stats['name']} (ID: {cluster_id})")
        print(f"  Total whistles: {stats['total_whistles']}")

        print(f"\n  Count per year N_k(y):")
        for year in unique_years:
            count = stats['count_per_year'][year]
            print(f"    {year}: {count}")

        print(f"\n  Year distribution p_k(y):")
        for year in unique_years:
            prob = stats['year_distribution'][year]
            print(f"    {year}: {prob:.4f} ({prob*100:.1f}%)")

        print(f"\n  Entropy of year distribution: {stats['year_entropy']:.4f} bits")

        # Determine if cluster is dominated by one year
        max_year_prob = max(stats['year_distribution'].values())
        if max_year_prob > 0.8:
            dominant_year = max(stats['year_distribution'], key=stats['year_distribution'].get)
            print(f"  → Dominated by year {dominant_year} ({max_year_prob*100:.1f}%)")

def plot_cluster_year_distributions(cluster_stats, unique_years, output_path):
    """Create visualization of year distributions across clusters."""
    # Filter out noise cluster for main plot
    cluster_ids = sorted([cid for cid in cluster_stats.keys() if cid != -1])

    if len(cluster_ids) == 0:
        print("No clusters found (only noise)")
        return

    # Create figure
    n_clusters = len(cluster_ids)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Stacked bar chart of year distributions
    ax1 = axes[0]
    width = 0.8
    x = np.arange(len(cluster_ids))

    # Create stacked bars
    bottom = np.zeros(len(cluster_ids))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_years)))

    for i, year in enumerate(unique_years):
        proportions = [cluster_stats[cid]['year_distribution'][year] for cid in cluster_ids]
        ax1.bar(x, proportions, width, bottom=bottom, label=str(year), color=colors[i])
        bottom += proportions

    ax1.set_xlabel("Cluster ID", fontsize=12)
    ax1.set_ylabel("Proportion of whistles", fontsize=12)
    ax1.set_title("Year Distribution per Cluster", fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(cid) for cid in cluster_ids])
    ax1.legend(title="Year", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Entropy per cluster
    ax2 = axes[1]
    entropies = [cluster_stats[cid]['year_entropy'] for cid in cluster_ids]
    colors_entropy = ['red' if e < 1.0 else 'orange' if e < 1.5 else 'green' for e in entropies]

    ax2.bar(x, entropies, width, color=colors_entropy, alpha=0.7, edgecolor='black')
    ax2.axhline(y=np.log2(len(unique_years)), color='blue', linestyle='--', linewidth=2,
                label=f'Max entropy ({np.log2(len(unique_years)):.2f} bits)')
    ax2.set_xlabel("Cluster ID", fontsize=12)
    ax2.set_ylabel("Entropy (bits)", fontsize=12)
    ax2.set_title("Year Distribution Entropy per Cluster\n(Lower = more dominated by one year)",
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(cid) for cid in cluster_ids])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    plt.show()

def main():
    data_dir = "../data"

    # Dataset files to load
    dataset_files = [
        "dataset_2019_2020.json",
        "dataset_2022.json",
        "dataset_2023.json",
        "dataset_2024_jan-apr.json"
    ]

    print("="*70)
    print("HDBSCAN CLUSTERING OF WHISTLE EMBEDDINGS")
    print("="*70)

    # Load data
    print("\nLoading embeddings and metadata...")
    embeddings, years, recording_names, labels = collect_embeddings_with_metadata(
        data_dir, dataset_files
    )

    print(f"\nTotal embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    unique_years = sorted(set(years))
    print(f"Years in dataset: {unique_years}")

    # Standardize embeddings
    print("\nStandardizing embeddings...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Run HDBSCAN clustering
    print("\nRunning HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=50,  # Minimum cluster size
        min_samples=10,       # Minimum samples in neighborhood
        metric='euclidean',
        cluster_selection_method='eom'  # Excess of Mass
    )
    cluster_labels = clusterer.fit_predict(embeddings_scaled)

    # Print clustering summary
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    print(f"\nClustering complete!")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Noise points: {n_noise} ({n_noise/len(cluster_labels)*100:.1f}%)")
    print(f"  Clustered points: {len(cluster_labels) - n_noise} ({(len(cluster_labels)-n_noise)/len(cluster_labels)*100:.1f}%)")

    # Compute cluster statistics
    print("\nComputing cluster statistics...")
    cluster_stats, unique_years = compute_cluster_statistics(cluster_labels, years)

    # Print statistics
    print_cluster_statistics(cluster_stats, unique_years)

    # Plot results
    print("\nGenerating plots...")
    plot_cluster_year_distributions(cluster_stats, unique_years, "../data/cluster_year_distributions.png")

    # Save results
    print("\nSaving results...")
    results = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "cluster_labels": cluster_labels,
        "cluster_stats": cluster_stats
    }

    np.savez("../data/cluster_results.npz",
             cluster_labels=cluster_labels,
             n_clusters=n_clusters,
             n_noise=n_noise)

    # Save detailed stats to JSON
    # Convert numpy types to Python types for JSON serialization
    cluster_stats_json = {}
    for cid, stats in cluster_stats.items():
        cluster_stats_json[str(cid)] = {
            "name": stats["name"],
            "total_whistles": int(stats["total_whistles"]),
            "count_per_year": {str(k): int(v) for k, v in stats["count_per_year"].items()},
            "year_distribution": {str(k): float(v) for k, v in stats["year_distribution"].items()},
            "year_entropy": float(stats["year_entropy"])
        }

    with open("../data/cluster_stats.json", "w") as f:
        json.dump(cluster_stats_json, f, indent=2)

    print("Results saved to:")
    print("  - ../data/cluster_results.npz")
    print("  - ../data/cluster_stats.json")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nFound {n_clusters} clusters in {len(embeddings)} embeddings")
    print(f"Years analyzed: {unique_years}")

    # Find clusters with lowest/highest entropy
    non_noise_stats = {cid: stats for cid, stats in cluster_stats.items() if cid != -1}
    if non_noise_stats:
        min_entropy_cluster = min(non_noise_stats.items(), key=lambda x: x[1]['year_entropy'])
        max_entropy_cluster = max(non_noise_stats.items(), key=lambda x: x[1]['year_entropy'])

        print(f"\nMost year-specific cluster (lowest entropy):")
        print(f"  {min_entropy_cluster[1]['name']}: {min_entropy_cluster[1]['year_entropy']:.4f} bits")

        print(f"\nMost year-diverse cluster (highest entropy):")
        print(f"  {max_entropy_cluster[1]['name']}: {max_entropy_cluster[1]['year_entropy']:.4f} bits")

if __name__ == "__main__":
    main()
