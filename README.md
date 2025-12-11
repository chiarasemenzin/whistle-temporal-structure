# Whistle Temporal Structure Analysis

This project analyzes temporal dependencies in dolphin whistle sequences by predicting the next whistle label based on context.

## Overview

The analysis evaluates whether longer temporal context improves prediction of the next whistle type in a bout:
- **Input**: Concatenated embeddings of k previous whistles
- **Output**: Label of the next whistle
- **Goal**: Determine if cross-entropy decreases as context length k increases

## Dataset Structure

```python
dataset = {
    "Recording_1": {
        "id": "Recording_1",
        "bouts": {
            "bout_1": {
                "whistle_1": {
                    "label": "...",
                    "embedding": [...],
                    "start_time": ...,
                    "end_time": ...,
                    ...
                },
                "whistle_2": { ... },
            },
            "bout_2": { ... },
        }
    },
    "Recording_2": { ... },
}
```

## Experimental Design

### Training
- For each context length k ∈ {2, 3, 4, 5, 6, 7}
- Use all bouts with length ≥ k+1
- Slide window of size k through each bout
- Creates thousands of training samples per k

### Testing
- Use only bouts with length ≥ 8
- Construct shared prediction positions across all k values
- Ensures fair comparison: all models evaluated on exact same positions

### Analysis
- Plot cross-entropy vs k
- Plot accuracy vs k
- **Flat line** → weak temporal dependence
- **Downward slope** → increasing context helps prediction

## Files

### Core Scripts
- `dataset_creation.py` - Creates dataset from raw recordings
- `prepare_training_data.py` - Prepares sliding window samples
- `train_and_evaluate.py` - Trains models and evaluates
- `run_experiment.py` - Complete pipeline runner

### Analysis Scripts
- `analyze_bout_lengths.py` - Statistics on bout lengths
- `visualize_embeddings.py` - UMAP visualization of embeddings

## Usage

### 1. Create Dataset
```bash
conda activate seq-dolphins
cd src
python dataset_creation.py
```

This creates `dataset.json` from recordings in the specified folder.

### 2. Run Complete Experiment
```bash
conda activate seq-dolphins
cd src
python run_experiment.py
```

This will:
1. Prepare training/test data with sliding windows
2. Train logistic regression for each k
3. Evaluate on shared test positions
4. Generate plots showing cross-entropy and accuracy vs k

### 3. Analyze Results
Output files in `data/`:
- `train_data.npz` - Training samples for all k values
- `test_data.npz` - Test samples with shared positions
- `evaluation_results.npz` - Cross-entropy and accuracy per k
- `context_length_evaluation.png` - Visualization of results

## Interpretation

The key metric is **cross-entropy loss vs context length**:

- **Decreasing cross-entropy**: Longer context provides useful information about the next whistle type → temporal structure exists
- **Flat cross-entropy**: Context length doesn't help → weak or no temporal dependencies
- **Increasing cross-entropy**: Longer context hurts (unusual, may indicate overfitting)

Secondary metric is **accuracy**, which provides interpretability but is less sensitive than cross-entropy.

## Requirements

```
numpy
scikit-learn
matplotlib
umap-learn
```

Install with:
```bash
conda activate seq-dolphins
pip install numpy scikit-learn matplotlib umap-learn
```
