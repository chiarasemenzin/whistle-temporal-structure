import json

def analyze_bout_lengths(dataset_path):
    """
    Analyze bout lengths in the dataset and count how many bouts
    are longer than N whistles for N in {1,2,3,4,5,6,7,8}.
    """
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # Count bout lengths
    bout_lengths = []

    for recording_name, recording_data in dataset.items():
        for bout_name, bout_data in recording_data["bouts"].items():
            bout_length = len(bout_data)  # number of whistles in the bout
            bout_lengths.append(bout_length)

    # Analyze for different N values
    N_values = [1, 2, 3, 4, 5, 6, 7, 8]

    print(f"Total number of bouts: {len(bout_lengths)}")
    print(f"Total number of recordings: {len(dataset)}")
    print()
    print("Bout length analysis:")
    print("-" * 50)

    for N in N_values:
        count = sum(1 for length in bout_lengths if length > N)
        percentage = (count / len(bout_lengths) * 100) if bout_lengths else 0
        print(f"Bouts with more than {N} whistles: {count} ({percentage:.1f}%)")

    print()
    print("Additional statistics:")
    print("-" * 50)
    print(f"Minimum bout length: {min(bout_lengths) if bout_lengths else 0}")
    print(f"Maximum bout length: {max(bout_lengths) if bout_lengths else 0}")
    print(f"Average bout length: {sum(bout_lengths) / len(bout_lengths):.2f}" if bout_lengths else 0)

    # Distribution of bout lengths
    print()
    print("Bout length distribution:")
    print("-" * 50)
    from collections import Counter
    length_counts = Counter(bout_lengths)
    for length in sorted(length_counts.keys())[:20]:  # Show first 20
        count = length_counts[length]
        percentage = (count / len(bout_lengths) * 100)
        print(f"Length {length}: {count} bouts ({percentage:.1f}%)")

    if len(length_counts) > 20:
        print(f"... and {len(length_counts) - 20} more length values")

if __name__ == "__main__":
    dataset_path = "../data/dataset.json"
    analyze_bout_lengths(dataset_path)
