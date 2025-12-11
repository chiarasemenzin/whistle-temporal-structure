import json
import numpy as np
import os

def load_embeddings_and_metadata(recording_id):
    with open(recording_id+"/window_embeddings_metadata.json", "r") as f:
        whistles = json.load(f)  # list of dicts
    embs = np.load(recording_id+"/window_embeddings_encoder.npy")  # shap
    for w in whistles:
        idx = w["index"]
        w["embedding"] = embs[idx]  # np.ndarray of shape (d,)
    return whistles

def build_bouts_for_recording(whistles, recording_id, gap=6.0):
    bouts = []
    current_bout = []

    for i, w in enumerate(whistles):
        if i == 0:
            current_bout = [w]
            continue

        prev = whistles[i - 1]
        gap_to_prev = w["start_time"] - prev["end_time"]

        if gap_to_prev <= gap: # same bout
            current_bout.append(w)
        else:
            # close previous bout and start a new one
            if current_bout:
                bouts.append(current_bout)
            current_bout = [w]
    # add last bout
    if current_bout:
        bouts.append(current_bout)

    # build the nested structure
    recording_struct = { "id": recording_id, "bouts": {}}

    for bout_idx, bout_whistles in enumerate(bouts, start=1):
        bout_name = f"bout_{bout_idx}"
        recording_struct["bouts"][bout_name] = {}

        for w_idx, w in enumerate(bout_whistles, start=1):
            whistle_name = f"whistle_{w_idx}"
            # you can keep only the fields you care about here
            recording_struct["bouts"][bout_name][whistle_name] = {
                "label":       w.get("label"),
                "start_time":  w.get("start_time"),
                "end_time":    w.get("end_time"),
                "embedding":   w.get("embedding"),  # your pooled embedding
                "index":       w.get("index"),
                "confidence":  w.get("confidence"),
                "duration":    w.get("end_time") - w.get("start_time"),
            }

    return recording_struct

def count_whistles_in_recording(rec_path):
    """Count the number of whistles in a recording."""
    json_file = os.path.join(rec_path, "window_embeddings_metadata.json")
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            whistles = json.load(f)
        return len(whistles)
    return 0

def create_dataset(rec_folder, gap=6.0, max_whistles=None):
    dataset = {}
    n = 0
    total_whistles = 0

    # Get list of recordings
    recordings = []
    for rec in os.listdir(rec_folder):
        rec_path = os.path.join(rec_folder, rec)
        if os.path.isdir(rec_path):
            json_file = os.path.join(rec_path, "window_embeddings_metadata.json")
            npy_file = os.path.join(rec_path, "window_embeddings_encoder.npy")

            if not os.path.exists(json_file) or not os.path.exists(npy_file):
                print(f"Skipping {rec_path}: missing required files")
                continue

            whistle_count = count_whistles_in_recording(rec_path)
            recordings.append((rec, rec_path, whistle_count))

    # If max_whistles is set, sample recordings
    if max_whistles is not None and len(recordings) > 0:
        # Sort by whistle count to get a diverse sample
        recordings.sort(key=lambda x: x[2])

        # Sample recordings until we reach approximately max_whistles
        sampled_recordings = []
        for rec_name, rec_path, whistle_count in recordings:
            if total_whistles + whistle_count > max_whistles and len(sampled_recordings) > 0:
                break
            sampled_recordings.append((rec_name, rec_path))
            total_whistles += whistle_count

        recordings = sampled_recordings
        print(f"Sampled {len(recordings)} recordings with ~{total_whistles} whistles (target: {max_whistles})")
    else:
        recordings = [(rec_name, rec_path) for rec_name, rec_path, _ in recordings]

    # Process selected recordings
    for rec_name, rec_path in recordings:
        n += 1
        print(f"Processing recording {n}/{len(recordings)}: {rec_path}")
        whistles = load_embeddings_and_metadata(rec_path)
        recording_struct = build_bouts_for_recording(whistles, rec_path, gap=gap)

        # Use the recording name as the key
        dataset[rec_name] = recording_struct

    print(f"Done! Processed {n} recordings with {total_whistles} whistles")
    return dataset


if __name__ == "__main__":
    base_folder = "/media/DOLPHIN1/new_extraction_dataset"

    # Define year folders in the correct order
    year_folders = [
        "end_2021",
        "start_2021",
    ]

    # Target number of whistles per year
    max_whistles_per_year = 20000

    for year_folder in year_folders:
        print(f"\n{'='*60}")
        print(f"Processing year: {year_folder}")
        print(f"{'='*60}")

        rec_folder = os.path.join(base_folder, year_folder)

        if not os.path.exists(rec_folder):
            print(f"Warning: Folder {rec_folder} does not exist. Skipping.")
            continue

        # Create dataset with sampling
        dataset = create_dataset(rec_folder, max_whistles=max_whistles_per_year)

        # Save the dataset
        output_filename = f"dataset_{year_folder}.json"
        with open(output_filename, "w") as f:
            json.dump(dataset, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

        print(f"Saved dataset to {output_filename}")

    print(f"\n{'='*60}")
    print("All datasets created successfully!")
    print(f"{'='*60}")
