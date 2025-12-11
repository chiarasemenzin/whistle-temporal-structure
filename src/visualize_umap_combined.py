import json
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
import os

def extract_channel_from_id(recording_id):
    """Extract channel information from recording ID.
    Assumes format like: .../channel_1 or .../channel_2 etc.
    Returns 'No Channel' for 2019_2020 data that doesn't have channel info.
    """
    import re
    # Look for pattern like "channel_1", "channel_2", etc.
    match = re.search(r'channel_(\d+)', recording_id)
    if match:
        return f"channel_{match.group(1)}"
    return "No Channel"

def load_embeddings_and_channels_from_dataset(dataset_file):
    """Load all embeddings and their channel labels from a dataset JSON file."""
    with open(dataset_file, "r") as f:
        dataset = json.load(f)

    embeddings = []
    channels = []

    for recording_name, recording_data in dataset.items():
        # Extract channel from the "id" field
        recording_id = recording_data.get("id", recording_name)
        channel = extract_channel_from_id(recording_id)

        for bout_name, bout_data in recording_data["bouts"].items():
            for whistle_name, whistle_data in bout_data.items():
                emb = np.array(whistle_data["embedding"])
                embeddings.append(emb)
                channels.append(channel)

    return np.array(embeddings), channels

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

    print("Loading embeddings from all years...")

    all_embeddings = []
    all_channels = []

    for year_folder in year_folders:
        dataset_file = f"../data/dataset_{year_folder}.json"

        if not os.path.exists(dataset_file):
            print(f"Warning: {dataset_file} not found. Skipping.")
            continue

        print(f"  Loading {year_folder}...")
        embeddings, channels = load_embeddings_and_channels_from_dataset(dataset_file)

        all_embeddings.append(embeddings)
        all_channels.extend(channels)

        print(f"    Loaded {len(embeddings)} embeddings")

    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    print(f"\nTotal embeddings: {len(all_embeddings)}")

    # Get unique channels and count
    unique_channels = sorted(set(all_channels))
    print(f"Found {len(unique_channels)} unique channels: {unique_channels}")

    # Count embeddings per channel
    channel_counts = {}
    for channel in all_channels:
        channel_counts[channel] = channel_counts.get(channel, 0) + 1

    print("\nChannel distribution:")
    for channel in unique_channels:
        print(f"  {channel}: {channel_counts[channel]} whistles")

    # Compute UMAP on all embeddings
    print("\nComputing UMAP on concatenated dataset...")
    umap_model = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_embeddings = umap_model.fit_transform(all_embeddings)

    # Create plot
    print("\nCreating visualization...")
    fig, ax = plt.subplots(figsize=(12, 9))

    # Define colors for channels (using a colormap for flexibility)
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab10')
    channel_colors = {channel: cmap(i % 10) for i, channel in enumerate(unique_channels)}

    # Plot each channel with its own color
    for channel in unique_channels:
        # Get indices for this channel
        indices = [i for i, ch in enumerate(all_channels) if ch == channel]

        ax.scatter(umap_embeddings[indices, 0],
                  umap_embeddings[indices, 1],
                  s=1, alpha=0.6, c=[channel_colors[channel]],
                  label=f"{channel} ({len(indices)})")

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title("UMAP Projection of Whistle Embeddings by Channel", fontsize=14, fontweight='bold')
    ax.legend(loc='best', markerscale=5, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = "umap_combined_by_channel.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to {output_file}")

    plt.show()

if __name__ == "__main__":
    main()
