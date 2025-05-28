# uvicorn API:app --reload
# http://127.0.0.1:8000/docs
import io
import librosa
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from pydantic import BaseModel
from model_cnn import GenreCNN
from typing import Optional, Tuple

# Configuration
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

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = GenreCNN(
    n_mels=N_MELS, n_genres=len(GENRES), clip_duration=CLIP_DUR,
    sr=SR, hop_length=HOP_LEN
)
cnn.load_state_dict(torch.load(CNN_PATH, map_location=device))
cnn.eval().to(device)

# FastAPI app
app = FastAPI(
    title="Music Genre Classifier API",
    version="1.1",
    description = (
        "Upload a short WAV audio clip only. "
        "The app will analyze the audio, extract mel spectrogram features, and "
        "classify it into one of 10 music genres: blues, classical, country, "
        "disco, hiphop, jazz, metal, pop, reggae, or rock."
    )
)


class PredictionResponse(BaseModel):
    genre: str
    confidence: Optional[float] = None


def read_audio_file(file_data: bytes) -> torch.Tensor:
    try:
        y, _ = librosa.load(io.BytesIO(file_data), sr=SR, duration=CLIP_DUR)
        if y is None or len(y) < 1:
            raise ValueError("Audio data is empty or corrupted.")
        mel = librosa.feature.melspectrogram(
            y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LEN
        )
        mel_db = librosa.power_to_db(mel, ref=mel.max())
        mel_norm = (
            mel_db - mel_db.min()
        ) / (mel_db.max() - mel_db.min() + 1e-6)
        return (
            torch.tensor(mel_norm, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Audio preprocessing failed: {str(e)}"
        )


def get_prediction(input_tensor: torch.Tensor) -> Tuple[str, float]:
    with torch.no_grad():
        logits = cnn(input_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        confidence, idx = torch.max(probs, dim=1)
        return GENRES[idx.item()], confidence.item()


@app.post(
    "/predictions",
    response_model=PredictionResponse,
    summary="Classify a music clip",
    response_description="Predicted music genre and optional confidence score"
)
async def predict_genre(
    file: UploadFile = File(
        ..., description="A WAV file up to 10 seconds in length"
    ),
    response: Response = None
):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400, detail="Only WAV files are supported."
        )

    try:
        file_data = await file.read()
        if not file_data:
            raise HTTPException(
                status_code=400, detail="Uploaded file is empty."
            )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"File read error: {str(e)}"
        )

    input_tensor = read_audio_file(file_data)
    genre, confidence = get_prediction(input_tensor)

    response.headers["Cache-Control"] = "no-store"

    return PredictionResponse(genre=genre, confidence=confidence)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("API:app", host="0.0.0.0", port=PORT, reload=True)
