import os
import shutil
import json
from glob import glob
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("once")

# GTZAN root
DATASET_DIR = "music_genre_classification/data/genres_original"
OUTPUT_DIR = "audio_split"
SPLITS = {"fold_1": 0.17, "fold_2": 0.17, "fold_3": 0.17, "fold_4": 0.17, "fold_5": 0.17, "test": 0.15}
SEED = 42

for split in SPLITS:
    for genre in os.listdir(DATASET_DIR):
        os.makedirs(
            os.path.join(OUTPUT_DIR, split, genre),
            exist_ok=True
        )

# ─── HOLD OUT TEST & COLLECT TRAIN+VAL ────────────────────────────────────────
all_paths = []   
all_labels = []  

for genre in os.listdir(DATASET_DIR):
    genre_src = os.path.join(DATASET_DIR, genre)
    wavs = sorted(f for f in os.listdir(genre_src) if f.lower().endswith(".wav"))
    if not wavs:
        print(f"Skipping {genre!r}: no .wav files")
        continue

    # 1) carve off test
    train_val, test = train_test_split(
        wavs,
        test_size=SPLITS["test"],
        random_state=42,
        shuffle=True
    )

    # copy test files
    for fname in test:
        src = os.path.join(genre_src, fname)
        dst = os.path.join(OUTPUT_DIR, "test", genre, fname)
        shutil.copyfile(src, dst)

    # record the rest for CV
    for fname in train_val:
        all_paths.append(os.path.join(genre, fname))
        all_labels.append(genre)

# ─── STRATIFIED K-FOLD ON train_val ───────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold_idx, (_, val_idx) in enumerate(skf.split(all_paths, all_labels), start=1):
    fold_name = f"fold_{fold_idx}"
    for i in val_idx:
        rel_path = all_paths[i]         
        src      = os.path.join(DATASET_DIR, rel_path)
        dst      = os.path.join(OUTPUT_DIR, fold_name, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)

print("5-fold + test split complete!")