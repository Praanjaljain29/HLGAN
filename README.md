# HLGAN: Hierarchical Local-Global Attention Network for MNIST

## Overview
HLGAN is a lightweight Transformer-style image classification model for handwritten digit recognition on MNIST.
It combines local attention for neighborhood feature extraction and global attention for full-image context modeling.

## Features
- Patch-based image tokenization
- Local followed by global self-attention
- CLS token for classification
- LayerNorm and Dropout for training stability
- Compact architecture suitable for MNIST-scale tasks

## Architecture Summary
1. Input image: 28x28 grayscale
2. Patch extraction: 7x7 patches (16 patches total)
3. Patch embedding to token dimension
4. Add CLS token and positional embeddings
5. Local self-attention (restricted neighborhood)
6. Global self-attention (all-token context)
7. Classification head to 10 digit classes

## Tech Stack
- Python
- PyTorch
- Torchvision
- NumPy
- scikit-learn

## Repository Structure
- `dataset.py`: MNIST dataloader and preprocessing
- `model.py`: HLGAN architecture
- `train.py`: training and evaluation pipeline
- `README.md`: detailed project notes
- `README2.md`: concise GitHub-ready README

## Dataset
The project uses MNIST and downloads it automatically through torchvision if not present.
Default data directory: `./data`

## Installation
```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:
```bash
pip install torch torchvision scikit-learn numpy
```

## Training and Evaluation
Run:
```bash
python train.py
```

Default configuration in `train.py`:
- Epochs: 15
- Batch size: 64
- Learning rate: 1e-3
- Optimizer: Adam
- Loss: CrossEntropyLoss

## Metrics Reported
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- Confusion matrix
- Per-class classification report

## Sample Result
Observed run (example):
- Accuracy: 96.03%
- Precision: 0.9605
- Recall: 0.9603
- F1-score: 0.9603
