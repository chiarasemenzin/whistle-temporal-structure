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

rec_folder="/Users/chiarasemenzin/DOLPHIN1/new_extraction_dataset/2019_2020/"

for rec in os.listdir(rec_folder):
    rec_path = os.path.join(rec_folder, rec)
    if os.path.isdir(rec_path):
        json_file = os.path.join(rec_path, "window_embeddings_metadata.json")
        npy_file = os.path.join(rec_path, "window_embeddings_encoder.npy")

        if not os.path.exists(json_file) or not os.path.exists(npy_file):
            print(f"Skipping {rec_path}: missing required files")
            continue

        print("Processing recording:", rec_path)
        whistles=load_embeddings_and_metadata(rec_path)
        build_bouts_for_recording(whistles,rec_path)
        print("Done!")