# Deepfake voice detection system using neural vocoders	
Overview : 
The Deepfake Voice Detection System is a deep learning-based solution designed to identify AI-generated synthetic speech by detecting artifacts introduced by neural vocoders. As deepfake technology becomes increasingly sophisticated, distinguishing between real and artificially generated human voices has become a critical challenge for security, media integrity, and digital forensics. This project addresses that challenge by leveraging RawNet2, a state-of-the-art raw waveform-based deep learning architecture, to classify audio samples as either genuine or synthetic.

What the Project Does : 
The system analyzes raw audio waveforms directly—without requiring handcrafted feature extraction—to detect subtle anomalies left behind by neural vocoders used to generate fake speech. Neural vocoders, such as WaveNet, WaveGlow, and HiFi-GAN, synthesize human-like speech but often leave identifiable artifacts in the waveform. The RawNet2 model is trained to recognize these tell-tale signs, enabling accurate detection of deepfake audio across different vocoder types and synthesis conditions.

Multi-Task Learning Approach : 
A key innovation in this project is the use of multi-task learning, where the model simultaneously performs two tasks: (1) binary classification of real versus fake voice samples, and (2) identification of the specific neural vocoder used to generate the synthetic audio. By learning to recognize both the authenticity of a sample and its likely source vocoder, the model develops a richer internal representation of vocoder-specific artifacts. This shared learning process significantly improves detection accuracy and makes the system more robust to unseen vocoders and varied synthesis conditions, as the model learns generalized artifact patterns rather than overfitting to a single generation method.

Tools and Technologies :
The project is implemented using Python and PyTorch, with RawNet2 serving as the core deep learning architecture. The pipeline includes audio processing components for loading, preprocessing, and augmenting raw waveform data, enabling the model to learn directly from time-domain audio signals. The use of raw waveforms eliminates the information loss that typically occurs during traditional feature extraction (e.g., MFCCs or spectrograms), allowing the model to capture fine-grained vocoder artifacts that are often imperceptible in transformed representations.

Impact and Applications : 
This system has practical applications in media forensics, content authenticity verification, and defense against audio-based misinformation. By combining raw waveform analysis with multi-task learning, the project demonstrates that vocoder-aware deepfake detection can achieve high accuracy while remaining robust to the growing diversity of speech synthesis technologies. The approach is modular and extensible, making it adaptable to new vocoders and emerging deepfake generation techniques.





# Deepfake Voice Detection System using Neural Vocoders

A deep learning system that detects AI-generated synthetic speech by identifying artifacts left behind by neural vocoders. Built with RawNet2 and PyTorch, the model operates directly on raw audio waveforms and uses multi-task learning to jointly perform real/fake classification and vocoder identification.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Results](#results)
- [License](#license)

---

## Overview

Deepfake voice generation has become increasingly realistic thanks to neural vocoders such as WaveNet, WaveGlow, and HiFi-GAN. These vocoders synthesize human-like speech but leave subtle artifacts in the waveform. This project leverages **RawNet2**, a raw-waveform deep learning architecture, to detect those artifacts and classify audio samples as genuine or synthetic.

Unlike traditional approaches that rely on handcrafted features (MFCCs, spectrograms), this system learns directly from time-domain audio signals, preserving fine-grained vocoder artifacts that are often lost during feature transformation.

## Features

- **Raw waveform input** — no manual feature extraction required
- **Multi-task learning** — simultaneous real/fake classification and vocoder identification
- **Robust detection** — generalizes across different neural vocoder types
- **PyTorch implementation** — modular and extensible codebase

## Project Structure

```
deepfake-voice-detection/
├── data/                  # Dataset directory (not included)
├── models/                # Saved model checkpoints
├── src/
│   ├── dataset.py         # Data loading and preprocessing
│   ├── model.py           # RawNet2 architecture
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   └── inference.py       # Single-file inference
├── requirements.txt
├── README.md
└── LICENSE
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchaudio
- numpy
- librosa
- scikit-learn

## Installation

```bash
git clone https://github.com/yourusername/deepfake-voice-detection.git
cd deepfake-voice-detection
pip install -r requirements.txt
```

## Usage

### Training

```bash
python src/train.py --data_dir ./data --epochs 50 --batch_size 32 --lr 0.0001
```

### Evaluation

```bash
python src/evaluate.py --checkpoint ./models/best_model.pth --data_dir ./data/test
```

### Inference on a single audio file

```bash
python src/inference.py --audio path/to/audio.wav --checkpoint ./models/best_model.pth
```

Output example:

```
Prediction: FAKE (confidence: 0.94)
Detected vocoder: HiFi-GAN
```

## Model Architecture

The system uses **RawNet2** as the backbone, which processes raw audio waveforms through a series of sinc-convolutional filters and residual blocks. Two task-specific heads are attached on top of the shared encoder:

- **Classification head** — binary output (real / fake)
- **Vocoder identification head** — multi-class output (one class per vocoder type)

The shared encoder is trained jointly on both tasks, encouraging the model to learn vocoder-specific artifact representations that improve overall detection accuracy.

## Dataset

The model is trained on a balanced dataset of real and synthetic speech samples generated using multiple neural vocoders. The dataset is not included in this repository due to size constraints. To prepare your own dataset, organize audio files as follows:

```
data/
├── real/
│   ├── sample_001.wav
│   └── ...
└── fake/
    ├── wavenet/
    ├── waveglow/
    └── hifigan/
```

## Results

| Metric | Score |
|--------|-------|
| Accuracy (real/fake) | XX.X% |
| Vocoder identification accuracy | XX.X% |
| EER | X.X% |

*(Replace with your actual evaluation numbers.)*

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

This version keeps your project description but adds the practical sections people expect in a README: installation, usage, structure, and results. Fill in the placeholder numbers and your GitHub URL, and it's ready to push.
