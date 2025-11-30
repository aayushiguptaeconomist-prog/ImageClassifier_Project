# Bicycle vs Motorcycle Image Classification

This repository contains code and models for classifying images as either a bicycle or a motorcycle. It explores multiple approaches: a custom CNN, transfer learning with MobileNetV2, and a hybrid workflow that combines pretrained feature extraction with a classical ML classifier (Random Forest).

## Overview

- Approaches: Custom CNN, Transfer Learning (MobileNetV2), Feature Extraction + Random Forest
- Goal: Achieve robust, accurate predictions on unseen images (bicycle vs motorcycle) while systematically comparing model performance, generalization, and per-class metrics.

## Dataset

- Two classes: `bicycle`, `motorcycle`
- Data directories: `Data/bicycle/`, `Data/motorcycle/`
- The main analysis notebook is here: 
- Cleaning steps used: `imghdr` (file-type validation), `cv2.imread()` for detecting corrupted files
- Input pipeline built with `tf.data.Dataset` (cacheing, batching, shuffling, prefetching)

## Preprocessing

- Resize images to model input size (e.g., 224x224)
- Rescale pixel values to [0, 1] or use model-specific preprocessing (MobileNetV2)

## Models

### Custom CNN (From Scratch)

- Data augmentation: random flips, rotations, and brightness adjustments (added it later)
- Architecture: 3 convolutional blocks → Flatten + Dense + Dropout → Softmax output
- Training notes: strong training accuracy but weaker validation/generalization
- Prediction results tried on 10 random images: All predicted incorrectly

### Transfer Learning — MobileNetV2

- Start with ImageNet weights, add a GAP → Dense → Softmax head
- Typical workflow: freeze backbone, train head, fine-tune last N layers
- Fine-tuning requires low learning rate (e.g., 1e-5) to avoid catastrophic forgetting
- Result (approx): ~70% validation accuracy

### Feature Extraction (MobileNetV2) + RandomForest - HYBRID

- Extract embeddings from pretrained MobileNetV2 (e.g., a 128-d or pooled feature vector)
- Train a `RandomForestClassifier` (scikit-learn) on those extracted features
- Result (approx): ~71% overall accuracy; can improve class-wise performance in some cases

## Performance Summary

| Model                      | Bicycle Accuracy | Motorcycle Accuracy | Overall  |
|---------------------------:|:----------------:|:-------------------:|:--------:|
| MobileNetV2 (Frozen)       | 73.5%            | 67.4%               | ~70.4%   |
| MobileNetV2 + RandomForest | 74.9%            | 67.4%               | ~71.1%   |

## Key Learnings

- Transfer learning stabilizes performance on small datasets.
- Classical ML on pretrained embeddings can match or exceed naive fine-tuning in some cases.
- Early stopping and dropout in dense layers helps mitigate overfitting.
- Fine-tuning needs very small learning rates and careful monitoring of validation metrics.

## Usage

Predict on a single image (using self-built CNN):

```
predict_image("path/to/image.jpg")
```

Predict on a single image (using MobileNetV2):

```
predict_image_tl("path/to/image.jpg")
```

Predict on a single image (using MobileNetV2 + RandomForest hybrid):

```
predict_image_rf("path/to/image.jpg")
```

## Requirements

- `tensorflow`
- `numpy`
- `matplotlib`
- `opencv-python`
- `scikit-learn`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Future Work

- Try EfficientNetB0 or other modern backbones
- Add detailed evaluation artifacts: confusion matrix, precision/recall, per-class F1
- Address class imbalance and add more diverse images

## Contact

For questions or collaboration, open an issue or contact the author via the repository.