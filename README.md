# CIFAR-10 Image Classification with Custom Neural Network (TensorFlow/Keras)

This project implements and evaluates a custom convolutional neural network (CNN) for image classification on the CIFAR-10 dataset using TensorFlow and Keras. The notebook covers the full workflow from data exploration and preprocessing to model design, training, evaluation, and visualization.

## Features

- **Data Loading & Exploration:** Loads CIFAR-10 dataset, displays sample images, and analyzes class distribution.
- **Preprocessing:** Normalizes image data and one-hot encodes labels.
- **Custom Model Architecture:** 
  - Uses custom intermediate blocks with attention-like weighting.
  - Includes batch normalization and dropout for regularization.
  - Ends with a global average pooling and dense output layer.
- **Training:** 
  - Data augmentation with `ImageDataGenerator`.
  - Early stopping and metrics tracking.
- **Evaluation:** 
  - Computes accuracy, precision, recall, and confusion matrix.
  - Generates classification report.
- **Visualization:** 
  - Plots training/validation loss and accuracy.
  - Visualizes confusion matrix and prediction confidence for test images.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn

Install dependencies with:
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Usage

1. Open and run the notebook `ecs7026p-ec24233-nn-dl-cifar.ipynb` in Jupyter or VS Code.
2. The notebook will:
    - Load and preprocess CIFAR-10 data.
    - Build and train a custom CNN model.
    - Evaluate model performance and visualize results.

## Notes

- The model uses a custom block structure with learned weighting of convolutional outputs for improved feature extraction.
- Training and evaluation metrics are visualized for better interpretability.
- The notebook is self-contained and can be adapted for other image classification tasks.

---
