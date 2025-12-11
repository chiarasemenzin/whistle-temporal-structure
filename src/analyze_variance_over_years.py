import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import re

def load_dataset(dataset_path):
    """Load the dataset from JSON."""
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    return dataset

def extract_year_from_recording(recording_name):
    """
    Extract year from recording name.

    Assumes recording names contain year information.
    Adjust pattern as needed for your data.
    """
    # Try to find a 4-digit year in the recording name
    match = re.search(r'(19|20)\d{2}', recording_name)
    if match:
        return int(match.group(0))
    return None

def collect_embeddings_with_years(dataset):
    """
    Collect all embeddings along with their year information.

    Returns:
        embeddings: List of embeddings (numpy arrays)
        years: List of years corresponding to each embedding
        recording_names: List of recording names
    """
    embeddings = []
    years = []
    recording_names = []
    labels = []

    for recording_name, recording_data in dataset.items():
        year = extract_year_from_recording(recording_name)

        if year is None:
            print(f"Warning: Could not extract year from {recording_name}, skipping")
            continue

        for bout_name, bout_data in recording_data["bouts"].items():
            for whistle_name, whistle_data in bout_data.items():
                embedding = whistle_data.get("embedding")
                label = whistle_data.get("label")

                if embedding is not None:
                    embeddings.append(np.array(embedding))
                    years.append(year)
                    recording_names.append(recording_name)
                    labels.append(label)

    return np.array(embeddings), np.array(years), recording_names, labels

def bootstrap_variance(embeddings_pca, n_bootstrap=1000, seed=42):
    """
    Bootstrap confidence intervals for total variance.

    Args:
        embeddings_pca: PCA-transformed embeddings for one year
        n_bootstrap: Number of bootstrap samples
        seed: Random seed

    Returns:
        mean, lower_ci, upper_ci: Bootstrap statistics
    """
    rng = np.random.RandomState(seed)
    n_samples = len(embeddings_pca)

    bootstrap_vars = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        sample = embeddings_pca[indices]

        # Compute total variance
        total_var = np.sum(np.var(sample, axis=0))
        bootstrap_vars.append(total_var)

    bootstrap_vars = np.array(bootstrap_vars)
    mean = np.mean(bootstrap_vars)
    lower_ci = np.percentile(bootstrap_vars, 2.5)
    upper_ci = np.percentile(bootstrap_vars, 97.5)

    return mean, lower_ci, upper_ci

def main():
    dataset_path = "../data/dataset.json"

    print("="*70)
    print("VARIANCE ANALYSIS ACROSS YEARS")
    print("="*70)

    print("\nLoading dataset...")
    dataset = load_dataset(dataset_path)

    print("Extracting embeddings and year information...")
    embeddings, years, recording_names, labels = collect_embeddings_with_years(dataset)

    print(f"\nTotal embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    unique_years = np.unique(years)
    print(f"Years in dataset: {unique_years}")
    print(f"Embeddings per year:")
    for year in unique_years:
        count = np.sum(years == year)
        print(f"  {year}: {count} embeddings")

    # Step 1: Standardize globally
    print("\n" + "="*70)
    print("STEP 1: Global standardization")
    print("="*70)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    print(f"Scaled embeddings: mean={np.mean(embeddings_scaled):.6f}, std={np.std(embeddings_scaled):.6f}")

    # Step 2: Fit PCA on all data
    print("\n" + "="*70)
    print("STEP 2: Fit PCA on combined data")
    print("="*70)
    pca = PCA(n_components=0.95, random_state=42)  # Keep 95% variance
    embeddings_pca = pca.fit_transform(embeddings_scaled)

    n_components = pca.n_components_
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"PCA components: {n_components}")
    print(f"Explained variance: {explained_var*100:.2f}%")
    print(f"Top 5 PC explained variance ratios: {pca.explained_variance_ratio_[:5]}")

    # Step 3: Compute total variance per year
    print("\n" + "="*70)
    print("STEP 3: Compute total variance per year")
    print("="*70)

    year_variances = {}
    year_ci = {}

    for year in unique_years:
        mask = years == year
        year_embeddings_pca = embeddings_pca[mask]

        # Compute total variance (sum of variances across all PCs)
        total_var = np.sum(np.var(year_embeddings_pca, axis=0))

        # Bootstrap confidence intervals
        mean_var, lower_ci, upper_ci = bootstrap_variance(year_embeddings_pca, n_bootstrap=1000)

        year_variances[year] = total_var
        year_ci[year] = (lower_ci, upper_ci)

        print(f"\nYear {year}:")
        print(f"  Total variance: {total_var:.4f}")
        print(f"  Bootstrap 95% CI: [{lower_ci:.4f}, {upper_ci:.4f}]")
        print(f"  N samples: {np.sum(mask)}")

    # Step 4: Optional normalization
    total_variance_all = np.sum(np.var(embeddings_pca, axis=0))
    print(f"\nTotal variance across all years: {total_variance_all:.4f}")

    year_variances_normalized = {
        year: var / total_variance_all
        for year, var in year_variances.items()
    }

    # Step 5: Statistical test - linear regression
    print("\n" + "="*70)
    print("STEP 4: Linear regression test")
    print("="*70)

    years_array = np.array(list(year_variances.keys()))
    variances_array = np.array(list(year_variances.values()))

    # Fit linear regression: variance = a + b * year
    slope, intercept, r_value, p_value, std_err = stats.linregress(years_array, variances_array)

    print(f"\nLinear regression: Variance = {intercept:.4f} + {slope:.4f} * year")
    print(f"  R²: {r_value**2:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Standard error: {std_err:.6f}")

    if p_value < 0.05:
        if slope < 0:
            print(f"\n✓ Significant CONTRACTION: Embedding space contracts over years (p < 0.05)")
        else:
            print(f"\n✓ Significant EXPANSION: Embedding space expands over years (p < 0.05)")
    else:
        print(f"\n✗ No significant trend: p-value = {p_value:.4f} (not significant)")

    # Step 6: Plot
    print("\n" + "="*70)
    print("STEP 5: Plotting results")
    print("="*70)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Absolute variance with CI
    sorted_years = sorted(year_variances.keys())
    variances = [year_variances[y] for y in sorted_years]
    ci_lower = [year_ci[y][0] for y in sorted_years]
    ci_upper = [year_ci[y][1] for y in sorted_years]
    errors_lower = [variances[i] - ci_lower[i] for i in range(len(sorted_years))]
    errors_upper = [ci_upper[i] - variances[i] for i in range(len(sorted_years))]

    ax1.errorbar(sorted_years, variances,
                yerr=[errors_lower, errors_upper],
                fmt='o-', capsize=5, markersize=10, linewidth=2)

    # Add regression line
    years_range = np.linspace(min(sorted_years), max(sorted_years), 100)
    regression_line = intercept + slope * years_range
    ax1.plot(years_range, regression_line, 'r--', alpha=0.7, linewidth=2,
            label=f'y = {intercept:.2f} + {slope:.4f}x\n$R^2$={r_value**2:.3f}, p={p_value:.4f}')

    ax1.set_xlabel("Year", fontsize=14)
    ax1.set_ylabel("Total Variance in PCA Space", fontsize=14)
    ax1.set_title("Embedding Space Volume Over Years\n(with 95% Bootstrap CI)", fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # Plot 2: Normalized variance
    variances_norm = [year_variances_normalized[y] for y in sorted_years]
    ax2.bar(sorted_years, variances_norm, width=0.6, alpha=0.7, edgecolor='black')
    ax2.set_xlabel("Year", fontsize=14)
    ax2.set_ylabel("Normalized Variance (fraction of total)", fontsize=14)
    ax2.set_title("Normalized Embedding Space Volume", fontsize=16)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(variances_norm) * 1.1)

    plt.tight_layout()
    plt.savefig("../data/variance_over_years.png", dpi=300, bbox_inches='tight')
    print("Plot saved to ../data/variance_over_years.png")
    plt.show()

    # Save results
    results = {
        "years": sorted_years,
        "variances": variances,
        "variances_normalized": variances_norm,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "regression_slope": slope,
        "regression_intercept": intercept,
        "regression_r2": r_value**2,
        "regression_pvalue": p_value,
        "n_components": n_components,
        "explained_variance": explained_var
    }

    np.savez("../data/variance_over_years_results.npz", **results)
    print("Results saved to ../data/variance_over_years_results.npz")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nNumber of years analyzed: {len(unique_years)}")
    print(f"PCA components used: {n_components} (capturing {explained_var*100:.1f}% variance)")
    print(f"\nVariance trend: slope = {slope:.6f} per year")

    if p_value < 0.05:
        trend = "contracting" if slope < 0 else "expanding"
        print(f"→ Embedding space is significantly {trend} over time (p = {p_value:.6f})")
    else:
        print(f"→ No significant temporal trend detected (p = {p_value:.4f})")

if __name__ == "__main__":
    main()
