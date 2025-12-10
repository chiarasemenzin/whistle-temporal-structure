"""
Complete pipeline for temporal structure analysis.

This script:
1. Prepares training and test data with sliding windows
2. Trains models for different context lengths
3. Evaluates on shared test positions
4. Plots cross-entropy and accuracy vs context length
"""

import subprocess
import sys

def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print("\n" + "="*70)
    print(f"RUNNING: {description}")
    print("="*70 + "\n")

    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False,
        text=True
    )

    if result.returncode != 0:
        print(f"\nERROR: {script_name} failed with exit code {result.returncode}")
        sys.exit(1)

    print(f"\n✓ {description} completed successfully")

def main():
    print("="*70)
    print("TEMPORAL STRUCTURE ANALYSIS PIPELINE")
    print("="*70)

    # Step 1: Prepare data
    run_script(
        "prepare_training_data.py",
        "Step 1: Preparing training and test data"
    )

    # Step 2: Train and evaluate
    run_script(
        "train_and_evaluate.py",
        "Step 2: Training models and evaluating"
    )

    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nGenerated files:")
    print("  - ../data/train_data.npz")
    print("  - ../data/test_data.npz")
    print("  - ../data/evaluation_results.npz")
    print("  - ../data/context_length_evaluation.png")

if __name__ == "__main__":
    main()
