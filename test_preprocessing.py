# test_preprocessing.py
import os, random, sys
from preprocessing import preprocess_audio_dataset

SPLIT_DIR = "audio_split/train"

if not os.path.isdir(SPLIT_DIR):
    print(f"❌ Error: missing {SPLIT_DIR!r}. Run split.py first.")
    sys.exit(1)

audio_files, labels = [], []
for genre in os.listdir(SPLIT_DIR):
    genre_folder = os.path.join(SPLIT_DIR, genre)
    wavs = [f for f in os.listdir(genre_folder) if f.endswith(".wav")]
    if not wavs:
        continue
    pick = random.choice(wavs)
    audio_files.append(os.path.join(genre_folder, pick))
    labels.append(genre)

X, y, scaler, pca = preprocess_audio_dataset(audio_files, labels)

print("✅ Processed clips:", X.shape[0])
print("✅ Feature dim   :", X.shape[1])
print("✅ Sample labels :", y)