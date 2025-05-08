# split.py
import os
import shutil
from sklearn.model_selection import train_test_split

# GTZAN root
DATASET_DIR = "music_genre_classification/data/genres_original"
OUTPUT_DIR  = "audio_split"
SPLITS      = {"train": 0.70, "val": 0.15, "test": 0.15}

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
    for split_label, subset in zip(["train","val","test"], [train, val, test]):
        for fname in subset:
            src = os.path.join(genre_src, fname)
            dst = os.path.join(OUTPUT_DIR, split_label, genre, fname)
            shutil.copyfile(src, dst)

print("Split complete!")