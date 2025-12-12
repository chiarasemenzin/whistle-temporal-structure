import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import re

def load_dataset(filepath):
    """
    Load a single dataset JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        dataset: Dictionary containing the dataset
    """
    with open(filepath, "r") as f:
        return json.load(f)

def extract_year_month_from_recording(recording_name):
    """
    Extract year and month from recording name.

    Recording formats:
    - Format 1: Exp_DD_Mon_YYYY_HHMMam/pm (2019-2020 data)
      Example: Exp_25_Dec_2019_0645am -> (2019, 12)
    - Format 2: Exp_DD_Mon_YYYY_HHMM_channel_N (2022-2024 data)
      Example: Exp_27_Mar_2022_1745_channel_1 -> (2022, 3)

    Args:
        recording_name: Name of the recording

    Returns:
        year_month_label: String like "2022-03"
        year_month_numeric: Numeric value (year + (month-1)/12)
    """
    # Month name to number mapping
    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
        "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
        "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }

    # Try pattern 1: Exp_DD_Mon_YYYY_HHMM_channel_N (newer format)
    pattern1 = r'Exp_(\d{1,2})_([A-Za-z]+)_(\d{4})_\d{4}_channel_\d+'
    match = re.search(pattern1, recording_name)

    if not match:
        # Try pattern 2: Exp_DD_Mon_YYYY_HHMMam/pm (older format)
        pattern2 = r'Exp_(\d{1,2})_([A-Za-z]+)_(\d{4})_\d{4}(am|pm)'
        match = re.search(pattern2, recording_name)

    if match:
        day = int(match.group(1))
        month_name = match.group(2)
        year = int(match.group(3))

        month = month_map.get(month_name)
        if month is None:
            return None, None

        # Create label (e.g., "2022-03")
        year_month_label = f"{year}-{month:02d}"

        # Create numeric value for regression
        # year + (month-1)/12 gives continuous time
        # e.g., 2022-03 -> 2022.167
        year_month_numeric = year + (month - 1) / 12.0

        return year_month_label, year_month_numeric

    return None, None

def collect_embeddings_with_months(data_dir, dataset_files):
    """
    Collect all embeddings along with their month information from multiple datasets.

    Args:
        data_dir: Directory containing dataset JSON files
        dataset_files: List of dataset filenames to load

    Returns:
        embeddings: Array of embeddings
        months: Array of numeric month values
        month_labels: List of month label strings
        recording_names: List of recording names
        labels: List of whistle labels
    """
    embeddings = []
    months = []
    month_labels = []
    recording_names = []
    labels = []

    skipped_recordings = 0

    for filename in dataset_files:
        filepath = f"{data_dir}/{filename}"
        print(f"\nLoading {filename}...")

        try:
            dataset = load_dataset(filepath)
            print(f"  Found {len(dataset)} recordings")
        except FileNotFoundError:
            print(f"  Warning: {filepath} not found, skipping")
            continue

        # Process each recording
        for recording_name, recording_data in dataset.items():
            # Extract year-month from recording name
            month_label, month_numeric = extract_year_month_from_recording(recording_name)

            if month_label is None:
                skipped_recordings += 1
                continue

            # Extract embeddings from all bouts
            for bout_name, bout_data in recording_data["bouts"].items():
                for whistle_name, whistle_data in bout_data.items():
                    embedding = whistle_data.get("embedding")
                    label = whistle_data.get("label")

                    if embedding is not None:
                        embeddings.append(np.array(embedding))
                        months.append(month_numeric)
                        month_labels.append(month_label)
                        recording_names.append(recording_name)
                        labels.append(label)

    if skipped_recordings > 0:
        print(f"\nWarning: Skipped {skipped_recordings} recordings (couldn't extract date)")

    return np.array(embeddings), np.array(months), month_labels, recording_names, labels

def bootstrap_variance(embeddings_pca, n_bootstrap=1000, seed=42):
    """
    Bootstrap confidence intervals for total variance.

    Args:
        embeddings_pca: PCA-transformed embeddings for one month
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
    data_dir = "../data"

    # Dataset files to load
    dataset_files = [
        "dataset_2019_2020.json",
        "dataset_end_2021.json",
        "dataset_start_2021.json",
        "dataset_2022.json",
        "dataset_2023.json",
        "dataset_2024_jan-apr.json"
    ]

    print("="*70)
    print("VARIANCE ANALYSIS ACROSS MONTHS")
    print("="*70)

    print("\nExtracting embeddings and month information...")
    embeddings, months, month_labels_list, recording_names, labels = collect_embeddings_with_months(
        data_dir, dataset_files
    )

    print(f"\nTotal embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    unique_months = np.unique(months)
    print(f"\nNumber of distinct months: {len(unique_months)}")

    # Get unique month labels (sorted)
    month_label_to_numeric = {}
    for i, m in enumerate(month_labels_list):
        if m not in month_label_to_numeric:
            month_label_to_numeric[m] = months[i]

    sorted_month_labels = sorted(month_label_to_numeric.keys(), key=lambda x: month_label_to_numeric[x])

    print(f"\nMonths in dataset:")
    for month_label in sorted_month_labels:
        month_numeric = month_label_to_numeric[month_label]
        count = np.sum(np.isclose(months, month_numeric))
        print(f"  {month_label}: {count} embeddings")

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

    # Step 3: Compute total variance per month
    print("\n" + "="*70)
    print("STEP 3: Compute total variance per month")
    print("="*70)

    month_variances = {}
    month_ci = {}

    for month_numeric in unique_months:
        mask = np.isclose(months, month_numeric)
        month_embeddings_pca = embeddings_pca[mask]

        # Compute total variance (sum of variances across all PCs)
        total_var = np.sum(np.var(month_embeddings_pca, axis=0))

        # Bootstrap confidence intervals
        mean_var, lower_ci, upper_ci = bootstrap_variance(month_embeddings_pca, n_bootstrap=1000)

        month_variances[month_numeric] = total_var
        month_ci[month_numeric] = (lower_ci, upper_ci)

        # Find corresponding label
        month_label = None
        for ml in sorted_month_labels:
            if np.isclose(month_label_to_numeric[ml], month_numeric):
                month_label = ml
                break

        print(f"\nMonth {month_label} ({month_numeric:.3f}):")
        print(f"  Total variance: {total_var:.4f}")
        print(f"  Bootstrap 95% CI: [{lower_ci:.4f}, {upper_ci:.4f}]")
        print(f"  N samples: {np.sum(mask)}")

    # Step 4: Optional normalization
    total_variance_all = np.sum(np.var(embeddings_pca, axis=0))
    print(f"\nTotal variance across all months: {total_variance_all:.4f}")

    month_variances_normalized = {
        month: var / total_variance_all
        for month, var in month_variances.items()
    }

    # Step 5: Statistical test - linear regression
    print("\n" + "="*70)
    print("STEP 4: Linear regression test")
    print("="*70)

    months_array = np.array(list(month_variances.keys()))
    variances_array = np.array(list(month_variances.values()))

    # Fit linear regression: variance = a + b * month
    slope, intercept, r_value, p_value, std_err = stats.linregress(months_array, variances_array)

    print(f"\nLinear regression: Variance = {intercept:.4f} + {slope:.4f} * month")
    print(f"  R²: {r_value**2:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Standard error: {std_err:.6f}")

    if p_value < 0.05:
        if slope < 0:
            print(f"\n✓ Significant CONTRACTION: Embedding space contracts over time (p < 0.05)")
        else:
            print(f"\n✓ Significant EXPANSION: Embedding space expands over time (p < 0.05)")
    else:
        print(f"\n✗ No significant trend: p-value = {p_value:.4f} (not significant)")

    # Step 6: Plot
    print("\n" + "="*70)
    print("STEP 5: Plotting results")
    print("="*70)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Absolute variance with CI
    sorted_months = sorted(month_variances.keys())
    variances = [month_variances[m] for m in sorted_months]
    ci_lower = [month_ci[m][0] for m in sorted_months]
    ci_upper = [month_ci[m][1] for m in sorted_months]
    errors_lower = [variances[i] - ci_lower[i] for i in range(len(sorted_months))]
    errors_upper = [ci_upper[i] - variances[i] for i in range(len(sorted_months))]

    # Create labels for x-axis (showing month labels)
    x_labels = []
    for month_numeric in sorted_months:
        for ml in sorted_month_labels:
            if np.isclose(month_label_to_numeric[ml], month_numeric):
                x_labels.append(ml)
                break

    ax1.errorbar(sorted_months, variances,
                yerr=[errors_lower, errors_upper],
                fmt='o-', capsize=5, markersize=8, linewidth=2)

    # Add regression line
    months_range = np.linspace(min(sorted_months), max(sorted_months), 100)
    regression_line = intercept + slope * months_range
    ax1.plot(months_range, regression_line, 'r--', alpha=0.7, linewidth=2,
            label=f'y = {intercept:.2f} + {slope:.4f}x\n$R^2$={r_value**2:.3f}, p={p_value:.4f}')

    ax1.set_xlabel("Month", fontsize=14)
    ax1.set_ylabel("Total Variance in PCA Space", fontsize=14)
    ax1.set_title("Embedding Space Volume Over Months\n(with 95% Bootstrap CI)", fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # Set x-ticks to show month labels (show every Nth label if too many)
    step = max(1, len(x_labels) // 15)  # Show at most 15 labels
    ax1.set_xticks([sorted_months[i] for i in range(0, len(sorted_months), step)])
    ax1.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), step)], rotation=45, ha='right')

    # Plot 2: Normalized variance
    variances_norm = [month_variances_normalized[m] for m in sorted_months]
    ax2.bar(range(len(sorted_months)), variances_norm, width=0.8, alpha=0.7, edgecolor='black')
    ax2.set_xlabel("Month", fontsize=14)
    ax2.set_ylabel("Normalized Variance (fraction of total)", fontsize=14)
    ax2.set_title("Normalized Embedding Space Volume", fontsize=16)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(variances_norm) * 1.1)
    ax2.set_xticks(range(0, len(x_labels), step))
    ax2.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), step)], rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig("../data/variance_over_months.png", dpi=300, bbox_inches='tight')
    print("Plot saved to ../data/variance_over_months.png")
    plt.show()

    # Save results
    results = {
        "months": sorted_months,
        "month_labels": x_labels,
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

    np.savez("../data/variance_over_months_results.npz", **results)
    print("Results saved to ../data/variance_over_months_results.npz")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nNumber of months analyzed: {len(unique_months)}")
    print(f"Month range: {x_labels[0]} to {x_labels[-1]}")
    print(f"PCA components used: {n_components} (capturing {explained_var*100:.1f}% variance)")
    print(f"\nVariance trend: slope = {slope:.6f} per month")

    if p_value < 0.05:
        trend = "contracting" if slope < 0 else "expanding"
        print(f"→ Embedding space is significantly {trend} over time (p = {p_value:.6f})")
    else:
        print(f"→ No significant temporal trend detected (p = {p_value:.4f})")

if __name__ == "__main__":
    main()
