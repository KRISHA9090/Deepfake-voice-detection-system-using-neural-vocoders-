from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from model import RawNet2Inspired

ROOT = Path(__file__).resolve().parent
SR = 16000
CLIP_SECONDS = 1.5
SAMPLES = int(SR * CLIP_SECONDS)

class AudioDataset(Dataset):
    def __init__(self, manifest):
        self.df = pd.read_csv(manifest)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio, sr = sf.read(ROOT / row["file"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            raise ValueError(f"{row['file']} has sample rate {sr}; expected {SR} Hz.")
        if len(audio) < SAMPLES:
            audio = np.pad(audio, (0, SAMPLES-len(audio)))
        else:
            audio = audio[:SAMPLES]
        return torch.from_numpy(audio).unsqueeze(0), torch.tensor(int(row["label"]))

def main():
    torch.manual_seed(7)
    ds = AudioDataset(ROOT / "data" / "manifest.csv")
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    model = RawNet2Inspired()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, 6):
        total_loss, correct, count = 0.0, 0, 0
        for x, y in loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            count += len(y)
        print(f"Epoch {epoch}/5 | loss={total_loss/count:.4f} | "
              f"training_accuracy={correct/count:.2%}")

    out = ROOT / "models" / "rawnet2_inspired_demo.pth"
    torch.save({"model_state_dict": model.state_dict(),
                "sample_rate": SR,
                "labels": {"0": "real_demo_proxy", "1": "fake_demo_proxy"},
                "warning": "Demo-only model trained on synthetic signal proxies, not real speech."}, out)
    print(f"\nSaved demo checkpoint: {out}")
    print("WARNING: This score is not evidence of real-world deepfake detection.")

if __name__ == "__main__":
    main()
