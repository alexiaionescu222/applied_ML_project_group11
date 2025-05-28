# uvicorn API:app --reload
# http://127.0.0.1:8000/docs

import io
import librosa
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from model_cnn import GenreCNN

# config
PORT = 8000
SR = 22050
CLIP_DUR = 10
N_MELS = 128
HOP_LEN = 512
GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]
CNN_PATH = "cnn_best.pth"

# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = GenreCNN(n_mels=N_MELS, n_genres=len(GENRES),
               clip_duration=CLIP_DUR, sr=SR, hop_length=HOP_LEN)
cnn.load_state_dict(torch.load(CNN_PATH, map_location=device))
cnn.eval().to(device)

app = FastAPI(
    title="Music Genre Classifier API",
    version="1.0",
    description="Upload a short WAV clip and get back a predicted genre."
)


class Prediction(BaseModel):
    genre: str


@app.post(
    "/predict", response_model=Prediction, summary="Classify a music clip"
)
async def predict(
    file: UploadFile = File(..., description="A WAV file ≤ 10s")
):
    # check file type
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400, detail="Only WAV files are supported."
        )

    # read file contents
    try:
        data = await file.read()
        if not data:
            raise HTTPException(
                status_code=400, detail="Uploaded file is empty."
            )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to read uploaded file: {e}"
        )

    # load audio
    try:
        y, _ = librosa.load(io.BytesIO(data), sr=SR, duration=CLIP_DUR)
        if y is None or len(y) < 1:
            raise HTTPException(
                status_code=400, detail="Audio data is empty or corrupted."
            )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Could not decode audio: {e}"
        )

    # generate mel spectrogram
    try:
        mel = librosa.feature.melspectrogram(
            y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LEN
            )
        mel_db = librosa.power_to_db(mel, ref=mel.max())
        mel_norm = (
                mel_db - mel_db.min()
                ) / (mel_db.max() - mel_db.min() + 1e-6)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate mel spectrogram: {e}"
        )

    # format input for model
    try:
        inp = (
            torch.tensor(mel_norm, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            logits = cnn(inp)
            idx = int(logits.argmax(dim=1).item())
            genre = GENRES[idx]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return Prediction(genre=genre)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=PORT, reload=True)
