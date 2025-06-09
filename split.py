import os
import shutil
import json
from glob import glob
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# GTZAN root
DATASET_DIR = "music_genre_classification/data/genres_original"
OUTPUT_DIR = "audio_split"
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
N_FOLDS = 5
SEED = 42

# split folders
for split in SPLITS:
    for genre in os.listdir(DATASET_DIR):
        os.makedirs(os.path.join(OUTPUT_DIR, split, genre), exist_ok=True)

# the 70/15/15 split per genre
for genre in os.listdir(DATASET_DIR):
    genre_src = os.path.join(DATASET_DIR, genre)
    wavs = [f for f in os.listdir(genre_src) if f.endswith(".wav")]

    if not wavs:
        print(f"Skipping {genre!r}: no .wav files found")
        continue

    # hold out test
    train_val, test = train_test_split(
        wavs, test_size=SPLITS["test"], random_state=42
    )
    # split train vs val
    val_frac = SPLITS["val"] / (SPLITS["train"] + SPLITS["val"])
    train, val = train_test_split(
        train_val, test_size=val_frac, random_state=42
    )

    # copy files into audio_split/{train,val,test}/{genre}/
    for split_label, subset in zip(
        ["train", "val", "test"], [train, val, test]
    ):
        for fname in subset:
            src = os.path.join(genre_src, fname)
            dst = os.path.join(OUTPUT_DIR, split_label, genre, fname)
            shutil.copyfile(src, dst)

print("Initial 70/15/15 split complete!")

# build cross-validation folds on the union of train+val (85%)
all_paths = []
all_labels = []

# walk through train and val directories
for split_label in ("train", "val"):
    for genre in os.listdir(os.path.join(OUTPUT_DIR, split_label)):
        folder = os.path.join(OUTPUT_DIR, split_label, genre)
        for wav_path in glob(os.path.join(folder, "*.wav")):
            # store path from OUTPUT_DIR, e.g. "train/blues/blues.00000.wav"
            rel_path = os.path.relpath(wav_path, OUTPUT_DIR)
            all_paths.append(rel_path)
            all_labels.append(genre)

# stratified k-fold on the 85%
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = {}

for fold_idx, (train_idx, val_idx) in enumerate(
    skf.split(all_paths, all_labels)
):
    train_list = [all_paths[i] for i in train_idx]
    val_list = [all_paths[i] for i in val_idx]
    folds[f"fold_{fold_idx}"] = {
        "train": train_list,
        "val": val_list
    }

# save the CV mapping
with open(os.path.join(OUTPUT_DIR, "cv_folds.json"), "w") as f:
    json.dump(folds, f, indent=2)

print(
    f"Saved {N_FOLDS}-fold cross-validation indices to "
    f"'{OUTPUT_DIR}/cv_folds.json'"
)
