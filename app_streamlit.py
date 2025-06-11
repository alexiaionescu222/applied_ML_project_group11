import io
import subprocess
import sys
from typing import List

import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
import torch

from model_cnn import GenreCNN

PIPELINE_STEPS: List[str] = [
    "split.py",
    "preprocessing.py",
    "testing_preprocessing.py",
    "train_baseline_knn.py",
    "train_cnn.py",
    "evaluate_test.py",
]

SR = 22_050
CLIP_DUR = 10
N_MELS = 128
HOP_LEN = 512
GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="Music-Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)


def run_script(script_name: str) -> str:
    """Run a Python script with the current interpreter."""
    try:
        res = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=True,
            text=True,
        )
        return res.stdout or "(script produced no stdout)"
    except subprocess.CalledProcessError as exc:
        return f"❌ Error while running {script_name}:\n\n{exc.stderr or exc}"


@st.cache_resource
def load_cnn_model() -> GenreCNN:
    """Load the trained CNN exactly once per session."""
    model = GenreCNN(
        n_mels=N_MELS,
        n_genres=len(GENRES),
        clip_duration=CLIP_DUR,
        sr=SR,
        hop_length=HOP_LEN,
    ).to(DEVICE)
    model.load_state_dict(torch.load("cnn_best.pth", map_location=DEVICE))
    model.eval()
    return model


section = st.sidebar.radio(
    "Select section",
    ("Run Pipeline", "Classify Audio"),
    horizontal=False,
)

# Fixed conditional
if section == "Run Pipeline":

    st.title("🔄 Machine-Learning Pipeline")

    st.markdown(
        "Click *Run All* to execute the full sequence, "
        "or choose an individual step below:"
    )

    if st.button("Run All Steps!", use_container_width=True):
        for step in PIPELINE_STEPS:
            st.subheader(f"▶  {step}")
            out = run_script(step)
            st.code(out)
        st.success("✅ All steps completed!")

    st.markdown("---")

    choice = st.selectbox("Pick an individual step:", PIPELINE_STEPS)
    if st.button("Run Selected Step"):
        st.subheader(f"->  {choice}")
        out = run_script(choice)
        st.code(out)

else:

    st.title("🎵 Music-Genre Classifier")

    st.write(
        "Upload a *≥10-second WAV* clip. "
        "The app shows its waveform, Mel-spectrogram, "
        "and the genre predicted by the trained CNN."
    )

    file = st.file_uploader("Choose a WAV file", type=["wav"])
    if not file:
        st.info("Upload a clip to get started.")
        st.stop()

    raw_bytes = file.read()
    st.subheader("Audio Playback")
    st.audio(raw_bytes, format="audio/wav")

    try:
        y, _ = librosa.load(
            io.BytesIO(raw_bytes), sr=SR, duration=CLIP_DUR, mono=True
        )
    except Exception as exc:
        st.error(f"Couldn't decode audio: {exc}")
        st.stop()

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LEN
    )
    mel_db = librosa.power_to_db(mel, ref=mel.max())
    mel_norm = (
        mel_db - mel_db.min()
    ) / (mel_db.max() - mel_db.min() + 1e-6)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Waveform")
        fig1, ax1 = plt.subplots(figsize=(6, 2))
        librosa.display.waveshow(y, sr=SR, ax=ax1)
        ax1.set(
            xlabel="Time (s)", ylabel="Amplitude", title="Time-Domain Signal"
        )
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        st.subheader("Mel-Spectrogram (dB)")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        img = librosa.display.specshow(
            mel_db,
            sr=SR,
            hop_length=HOP_LEN,
            x_axis="time",
            y_axis="mel",
            cmap="magma",
            ax=ax2,
        )
        ax2.set(title="Mel-Spectrogram (dB)")
        fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
        st.pyplot(fig2)
        plt.close(fig2)

    cnn = load_cnn_model()
    with torch.no_grad():
        inp = (
            torch.tensor(mel_norm, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(DEVICE)
        )
        logits = cnn(inp)
        pred_idx = int(logits.argmax(dim=1).item())
        pred_genre = GENRES[pred_idx]

    st.markdown("---")
    st.subheader("Predicted Genre")
    st.success(f"*{pred_genre.upper()}*")