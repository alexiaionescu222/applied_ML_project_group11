# run_pipeline.py
import subprocess
import sys

def run(script_name):
    print(f"\n=== Running {script_name} ===")
    # sys.executable ensures we use the same Python interpreter (and env)
    subprocess.run([sys.executable, script_name], check=True)

if __name__ == "__main__":
    steps = [
        "split.py",
        "preprocessing.py",
        "test_preprocessing.py",
        "train_baseline_knn.py",
        "train_cnn.py",
        "evaluate_test.py"
    ]
    for step in steps:
        run(step)

    print("\n All steps finished successfully!")