# 🗺️ Landmark Image Classifier with MobileNetV2 + Albumentations

This project is a deep learning pipeline built to classify landmark images using a transfer learning approach with **MobileNetV2**, enhanced by **Albumentations**-based data augmentation. It includes training, evaluation, and prediction modules, and runs efficiently on GPU (if available).

---

## 📌 Features

- ✅ Uses **MobileNetV2** pretrained on ImageNet
- ✅ Custom image augmentation with **Albumentations**
- ✅ Organized using Keras `ImageDataGenerator` + custom data loader
- ✅ TensorBoard support for training visualization
- ✅ GPU/CPU compatible with TensorFlow and PyTorch checks
- ✅ Final model exported as `.h5` and ready for deployment

---

## 🧱 Directory Structure

LandMark classifier/
├── train/ # Training image folders (1 subfolder per class)
├── val/ # Validation image folders (same structure)
├── test/ # Testing images (unlabeled, per-class subfolders)
├── landmark_classifier.h5 # Final saved Keras model
├── logs/ # TensorBoard logs
└── main.py # Training and evaluation script

---

## 🧪 Model Architecture

- **Backbone**: MobileNetV2 (frozen for transfer learning)
- **Head**:
  - `GlobalAveragePooling2D`
  - `Dense(64, ReLU)`
  - `Dropout(0.3)`
  - `Dense(num_classes, Softmax)`

---

## 🛠️ Installation

```bash
# Create environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
