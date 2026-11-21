from pathlib import Path
import argparse
import numpy as np
import soundfile as sf
import torch
from model import RawNet2Inspired

SR = 16000
SAMPLES = int(1.5 * SR)
ROOT = Path(__file__).resolve().parent

def load_audio(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        raise ValueError(f"Expected 16 kHz audio; got {sr} Hz. Resample first.")
    if len(audio) < SAMPLES:
        audio = np.pad(audio, (0, SAMPLES-len(audio)))
    else:
        audio = audio[:SAMPLES]
    return torch.from_numpy(audio).view(1, 1, -1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="Path to a 16 kHz WAV file")
    args = parser.parse_args()
    ckpt = torch.load(ROOT/"models"/"rawnet2_inspired_demo.pth",
                      map_location="cpu", weights_only=False)
    model = RawNet2Inspired()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(load_audio(args.audio)), dim=1)[0]
    print(f"Demo proxy class: {'FAKE-like' if probs[1] > probs[0] else 'REAL-like'}")
    print(f"Class scores: real_proxy={probs[0]:.3f}, fake_proxy={probs[1]:.3f}")
    print("These scores are for synthetic demo signals only, not real voice authentication.")

if __name__ == "__main__":
    main()
