# Pothole Detection using Transfer Learning (MobileNetV2)

## Overview

This project implements a binary image classification system to detect potholes in road images using deep learning.

The model takes an input image and predicts:

- `pothole`
- `no_pothole`

The system uses transfer learning with MobileNetV2 and is optimized to run on Apple Silicon (M-series) using MPS acceleration.

---

## Project Structure

pothole-detection/
│
├── dataset/
│ ├── train/
│ │ ├── pothole/
│ │ └── no_pothole/
│ ├── val/
│ │ ├── pothole/
│ │ └── no_pothole/
│ └── test/
│ ├── pothole/
│ └── no_pothole/
│
├── train.py
├── inference.py
├── utils.py
├── requirements.txt
└── README.md


---

## Dataset

The dataset consists of road images categorized into two classes:

- `pothole`
- `no_pothole`

Data was split into:

- 70% Training
- 15% Validation
- 15% Test

Data augmentation is applied only to training images.

---

## Model Architecture

Model: MobileNetV2 (Pretrained on ImageNet)

Why MobileNetV2?

- Lightweight
- Efficient
- Suitable for edge deployment
- Good performance on limited datasets

Transfer learning approach:

1. Load pretrained weights
2. Freeze base layers
3. Replace final classification layer with 2 output neurons
4. Train classifier on pothole dataset

---

## Training Details

- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Learning Rate: 0.001
- Epochs: 10
- Image Size: 224x224
- Augmentations:
  - RandomHorizontalFlip
  - RandomRotation
  - RandomPerspective
  - GaussianBlur
  - ColorJitter

---

## Evaluation Metrics

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Example confusion matrix format:

[[TN FP]
[FN TP]]


Where:

- TN = True Negatives
- FP = False Positives
- FN = False Negatives
- TP = True Positives

---

## Installation

### 1. Clone Repository

git clone <your-repo-url>
cd pothole-detection


### 2. Create Virtual Environment

python -m venv venv
source venv/bin/activate


### 3. Install Dependencies

pip install -r requirements.txt


---

## Training the Model

Run:

python train.py


This will:

- Train the model
- Validate each epoch
- Evaluate on test set
- Save model as:

pothole_model.pth


---

## Running Inference

Place an image in the root directory.

Run:

python inference.py image.jpg


Output:

Prediction: pothole
Confidence: 0.93


---

## Hardware

Tested on:

- MacBook M-series (Apple Silicon)
- MPS acceleration enabled

Check MPS availability:

import torch
print(torch.backends.mps.is_available())


---

## Current Limitations

- Small test dataset may cause unstable accuracy
- Classification only (no bounding box detection)
- Performance depends heavily on dataset diversity

---

## Future Improvements

- Increase dataset size
- Fine-tune deeper layers
- Implement Grad-CAM visualization
- Upgrade to object detection (YOLO)
- Deploy as API (FastAPI)
- Convert to CoreML for iOS deployment

---

## Key Learning Points

- Transfer learning significantly reduces training time
- Data diversity impacts performance more than model size
- Proper train/val/test splitting prevents data leakage
- Small test sets produce misleading accuracy

---

## License

For educational and research purposes.
