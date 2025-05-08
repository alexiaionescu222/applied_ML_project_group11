# API.py
from flask import Flask, request, jsonify
import librosa
import torch
from model_cnn import GenreCNN
from collections import Counter

# configuration
PORT       = 8000
SR         = 22050
N_MELS     = 128
CLIP_DUR   = 10        # seconds
GENRES     = [
    "blues","classical","country","disco","hiphop",
    "jazz","metal","pop","reggae","rock"
]
CNN_PATH    = "cnn_best.pth"

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn    = GenreCNN(n_mels=N_MELS, n_genres=len(GENRES)).to(device)
cnn.load_state_dict(torch.load(CNN_PATH, map_location=device))
cnn.eval()

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]

    votes = []
    # Run inference on each 10 s slice of the 30 s track
    for offset in [0, CLIP_DUR, 2 * CLIP_DUR]:
        # load exactly one 10 s slice
        y, _ = librosa.load(f, sr=SR, offset=offset, duration=CLIP_DUR)

        # mel-spectrogram → dB → normalize to [0,1]
        mel     = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
        mel_db  = librosa.power_to_db(mel, ref=mel.max())
        mel_norm = (mel_db - mel_db.min())/(mel_db.max() - mel_db.min() + 1e-6)

        # to tensor shape [1,1,n_mels,time]
        inp = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # predict
        with torch.no_grad():
            logits = cnn(inp)
            idx    = logits.argmax(dim=1).item()
            votes.append(GENRES[idx])

        # rewind file pointer so next librosa.load starts at beginning
        f.stream.seek(0)

    # majority vote
    genre = Counter(votes).most_common(1)[0][0]
    return jsonify({"genre": genre})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)