![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![CNN](https://img.shields.io/badge/CNN-Image%20Classification-green)

# Skin Cancer Detection using CNN (MobileNetV2 Transfer Learning)

A Deep Learning-based image classification project that detects whether a skin lesion is **Malignant** or **Benign** using Convolutional Neural Networks and transfer learning.

---

## Problem Statement

Skin cancer is one of the most common types of cancer worldwide. Early and accurate detection plays a critical role in improving survival rates. 

This project aims to build a robust image classification model using deep learning to automatically classify dermoscopic skin images as:

- Malignant (Cancerous)
- Benign (Non-cancerous)

---

## Objective

- Build a CNN-based classifier using transfer learning
- Improve performance using fine-tuning
- Achieve high validation accuracy
- Develop a reproducible deep learning pipeline

---

## Dataset

This project uses the **Skin Cancer: Malignant vs Benign** dataset from Kaggle: https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign

- Binary classification dataset
- Dermoscopic skin lesion images
- Separate train and test folders

⚠ Due to size limitations, the dataset is not included in this repository.

To use the dataset:
1. Download it from Kaggle
2. Extract into a folder named `dataset/`
3. Update dataset path if needed

---

## Model Architecture

This project uses **MobileNetV2** as a pretrained base model. MobileNetV2 is a lightweight deep convolutional neural network pretrained on ImageNet. We use transfer learning to leverage pretrained weights and reduce training time.

### Why MobileNetV2?

- Lightweight and efficient
- Pretrained on ImageNet
- Good performance on image classification tasks
- Suitable for transfer learning

### Algorithm and Training Process

1. Import necessary libraries 
2. Load the dataset
3. Preprocess the images:
   - Resize to 224×224 pixels
   - Rescale pixel values to [0,1]
   - Apply data augmentation (rotation, flipping, zooming) 
4. Split dataset into Training (80%) and Validation (20%) sets
5. Load MobileNetV2 as the base model
6. Freeze base model layers to retain learned features
7. Add custom layers:
   - Global Average Pooling
   - Dropout (0.4)
   - Dense layer (1 neuron, sigmoid activation) 
8. Compile the model using Adam optimizer and Binary Cross-Entropy loss
9. Train the model for 15 epochs + 5 fine-tuning epochs
10. Evaluate the model on the test data and generate performance metrics (accuracy, confusion matrix, classification report)

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Transfer Learning

---

## Model Performance

- Classification Type: Binary (Malignant vs Benign)
- Image Size: 224x224
- Training Strategy: Transfer Learning + Fine-Tuning
- Final Model File: `skin_cancer_cnn_model.h5`
- Test Accuracy: 78.03%
- Training Accuracy: Improved steadily across epochs.
- Validation Accuracy: Showed minimal overfitting.
- Loss: Decreased smoothly with epochs.
- Confusion Matrix: Showed strong diagonal dominance, indicating correct predictions.

---

## Project Structure

```
skin-cancer-detection/
│
├── Skin_Cancer_Detection_CNN.ipynb
├── skin_cancer_model.h5
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### 1️) Clone Repository

```bash
git clone https://github.com/your-username/skin-cancer-detection-cnn.git
cd skin-cancer-detection-cnn
```

### 2) Create Virtual Environment

```bash
python -m venv venv
```

#### Activate:

#### Windows
```bash
venv\Scripts\activate
```

#### Mac/Linux
```bash
source venv/bin/activate
```

### 3) Install Dependencies
```bash
pip install -r requirements.txt
```

### 4) Download Dataset
- Download dataset from kaggle and extract it inside: ```dataset/```

### 5) Run Notebook
- Open: ```Skin_Cancer_Detection_CNN.ipynb```
- Run all cells to train or evaluate the model.

### Sample Prediction
- The trained model can be loaded using:

```python
from tensorflow.keras.models import load_model
model = load_model("skin_cancer_model.h5")
```

- You can then preprocess an image and perform predictions.

---

## Research Documentation

The detailed research paper explaining methodology, training strategy, and evaluation results is available in: ```research_paper/Skin_Cancer_Detection_Research_Paper.pdf```

---
## Future Improvements
- Convert into Flask web application
- Deploy using Streamlit
- Expand to multi-class skin disease detection
