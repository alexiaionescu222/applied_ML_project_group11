import os

print("\n=== Running split.py ===")
os.system("python split.py")
print("Audio split complete!\n")

print("=== Running preprocessing.py ===")
os.system("python preprocessing.py")
print("Preprocessing complete!\n")

print("=== Running train_baseline_knn.py ===")
os.system("python train_baseline_knn.py")
print("Baseline k-NN training complete!\n")

print("=== Running train_cnn.py ===")
os.system("python train_cnn.py")
print("CNN training complete!\n")

print("=== Running evaluate_test.py ===")
os.system("python evaluate_test.py")
print("Test set evaluation complete!\n")

print("All steps completed.")
