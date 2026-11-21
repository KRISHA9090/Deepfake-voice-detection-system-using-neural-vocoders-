from pathlib import Path
import tempfile
import streamlit as st
import numpy as np
import soundfile as sf
import torch
from model import RawNet2Inspired

ROOT = Path(__file__).resolve().parent
SR = 16000
SAMPLES = int(1.5 * SR)

st.set_page_config(page_title="Voice Authenticity Demo", page_icon="🎙️")
st.title("Voice Authenticity — RawNet2-inspired Demo")
st.warning("Educational demo only. The bundled model is trained on synthetic signal proxies, not real human or vocoder-generated speech. Do not use its output to judge whether a real recording is a deepfake.")

ckpt_path = ROOT / "models" / "rawnet2_inspired_demo.pth"
uploaded = st.file_uploader("Upload a 16 kHz WAV file", type=["wav"])

if not ckpt_path.exists():
    st.info("First run `python train.py` in the project terminal to create the demo checkpoint.")
elif uploaded:
    audio, sr = sf.read(uploaded, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        st.error(f"Please provide 16 kHz WAV audio. This file is {sr} Hz.")
    else:
        audio = audio[:SAMPLES] if len(audio) >= SAMPLES else np.pad(audio, (0, SAMPLES-len(audio)))
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = RawNet2Inspired()
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(torch.from_numpy(audio).view(1,1,-1)), dim=1)[0]
        st.metric("Demo proxy output", "FAKE-like" if probs[1] > probs[0] else "REAL-like")
        st.write({"real_proxy_score": round(float(probs[0]), 4),
                  "fake_proxy_score": round(float(probs[1]), 4)})
        st.caption("These are model scores on a toy proxy task, not calibrated deepfake probabilities.")
