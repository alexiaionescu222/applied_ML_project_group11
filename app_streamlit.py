# app_streamlit.py
# streamlit run app_streamlit.py
# http://localhost:8501


import streamlit as st

st.set_page_config(
    page_title="🎵 Music Genre Classifier",
    page_icon="🎶",
    layout="wide",
    initial_sidebar_state="expanded",
)
import io
import numpy as np
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
from model_cnn import GenreCNN

SR         = 22050
CLIP_DUR   = 10       # seconds
N_MELS     = 128
HOP_LEN    = 512
GENRES     = [
    "blues","classical","country","disco","hiphop",
    "jazz","metal","pop","reggae","rock"
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_cnn_model():
    model = GenreCNN(
        n_mels=N_MELS,
        n_genres=len(GENRES),
        clip_duration=CLIP_DUR,
        sr=SR,
        hop_length=HOP_LEN
    ).to(DEVICE)
    model.load_state_dict(torch.load("cnn_best.pth", map_location=DEVICE))
    model.eval()
    return model

cnn = load_cnn_model()

st.title("🎵 Music Genre Classifier")
st.write(
    """
    Upload a WAV clip,  
    visualize its waveform & Mel-spectrogram,  
    and see which of the 10 GTZAN genres the CNN predicts.
    """
)

uploaded = st.file_uploader("Choose a WAV file", type=["wav"])
if not uploaded:
    st.info("🎧 Upload a track snippet to get started.")
    st.stop()

raw_bytes = uploaded.read()

st.subheader("🔊 Audio Playback")
st.audio(raw_bytes, format="audio/wav")  

try:
    y, sr = librosa.load(io.BytesIO(raw_bytes), sr=SR, duration=CLIP_DUR)
except Exception as e:
    st.error(f"❌ Could not decode audio: {e}")
    st.stop()

mel = librosa.feature.melspectrogram(
    y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LEN
)
mel_db   = librosa.power_to_db(mel, ref=mel.max())
mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Waveform")
    fig_wav, ax_wav = plt.subplots(figsize=(6, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax_wav)
    ax_wav.set(title="Time Domain", xlabel="Time (s)", ylabel="Amplitude")
    st.pyplot(fig_wav)
    plt.close(fig_wav)

with col2:
    st.subheader("Mel-Spectrogram (dB)")
    fig_spec, ax_spec = plt.subplots(figsize=(6, 3))
    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=HOP_LEN,
        x_axis="time",
        y_axis="mel",
        ax=ax_spec,
        cmap="magma"
    )
    ax_spec.set(title="Mel-Spectrogram (dB)")
    fig_spec.colorbar(img, ax=ax_spec, format="%+2.0f dB")
    st.pyplot(fig_spec)
    plt.close(fig_spec)

inp = (
    torch.tensor(mel_norm, dtype=torch.float32)
         .unsqueeze(0).unsqueeze(0)
         .to(DEVICE)
)
with torch.no_grad():
    logits = cnn(inp)
    pred_idx   = int(logits.argmax(dim=1).item())
    pred_genre = GENRES[pred_idx]
    
st.markdown("---")
st.subheader("🎯 Predicted Genre")
st.success(f"**{pred_genre.upper()}**", icon="✅")
