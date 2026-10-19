# Lung Cancer Detection using Convolutional Neural Network (CNN)

A deep learning project that classifies lung histopathology images into three categories using a CNN built with TensorFlow/Keras.

## Overview

Computer vision is widely used in medical imaging, including detecting cancerous cells. This project builds a CNN classifier that distinguishes normal lung tissue from two types of lung cancer.

**Classes**

| Label | Description |
|-------|-------------|
| `lung_n` | Normal lung tissue |
| `lung_aca` | Lung adenocarcinoma |
| `lung_scc` | Lung squamous cell carcinoma |

## Dataset

- A custom subset of roughly 2,400 images across the three classes above.
- Provided as `lung_subset_small_folder.zip`, which extracts to a `lung_subset_small/` folder with one sub-folder per class:

```
lung_subset_small/
├── lung_n/
├── lung_aca/
└── lung_scc/
```

- Split: 80% training / 20% validation (via `ImageDataGenerator`).

## Tech Stack

- **Python 3**
- **TensorFlow / Keras**: model building and training
- **OpenCV, Pillow**: image processing
- **NumPy, Pandas**: data handling
- **Matplotlib**: visualization
- **Scikit-learn**: evaluation metrics

## Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>

pip install tensorflow opencv-python pillow numpy pandas matplotlib scikit-learn
```

## Project Workflow

1. **Import libraries**
2. **Extract the dataset** from the zip file
3. **Visualize** random samples from each class
4. **Prepare data** with `ImageDataGenerator` (rescaling to 1/255, 80/20 split, batch loading to avoid memory overflow)
5. **Build the CNN**
6. **Compile** with Adam optimizer and categorical cross-entropy loss
7. **Train** with callbacks
8. **Visualize** training vs. validation accuracy
9. **Evaluate** using a classification report

## Configuration

| Parameter | Value |
|-----------|-------|
| Image size | 128 × 128 |
| Batch size | 16 |
| Epochs | 10 |
| Validation split | 0.2 |

## Model Architecture

```
Conv2D(32, 5x5, ReLU, same) -> MaxPooling2D(2x2)
Conv2D(64, 3x3, ReLU, same) -> MaxPooling2D(2x2)
Conv2D(128, 3x3, ReLU, same) -> MaxPooling2D(2x2)
Flatten
Dense(256, ReLU) -> BatchNormalization
Dense(128, ReLU) -> Dropout(0.3) -> BatchNormalization
Dense(3, Softmax)
```

## Training Details

- **Optimizer:** Adam
- **Loss:** Categorical cross-entropy
- **Callbacks:**
  - `EarlyStopping`: monitors `val_accuracy`, patience 3, restores best weights
  - `ReduceLROnPlateau`: monitors `val_loss`, patience 2, factor 0.5
  - Custom `myCallback`: stops training once validation accuracy exceeds 90%

## Usage

Run the notebook or script step by step:

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zipfile import ZipFile
from PIL import Image
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
```

Then extract the data, build the generators, train the model and evaluate it as described in the workflow above.

## Evaluation

The model is evaluated on the validation set using `sklearn.metrics.classification_report` (precision, recall, F1-score per class).

> **Note:** Training accuracy approaches 1.0, which indicates some overfitting on this small dataset. Consider data augmentation, more data, or a stronger regularization strategy to improve generalization.

### Evaluation tips

- `flow_from_directory` orders classes **alphabetically** (`lung_aca`, `lung_n`, `lung_scc`). Use `list(val_data.class_indices.keys())` as `target_names` so the report labels match.
- Create the validation generator with `shuffle=False` so predictions line up with `val_data.classes`.

## Possible Improvements

- Add data augmentation (rotation, flips, zoom)
- Use transfer learning (e.g., VGG16, ResNet50, EfficientNet)
- Train on the full dataset with higher resolution
- Add a confusion matrix and per-class analysis
- Save and load the trained model for inference

## Disclaimer

This project is for **educational and research purposes only**. It is not a medical device and must not be used for clinical diagnosis.

## License

Add your preferred license here (e.g., MIT).
