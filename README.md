# 😃 Facial Emotion Detection (FED) using CNNs

## 🧠 Introduction

**Facial Emotion Detection (FED)** refers to the automatic recognition of human emotions through the analysis of facial expressions using **Computer Vision** and **Machine Learning**. It is increasingly relevant in domains like:

- 🎓 Education (student engagement)
- 🧠 Psychology (behavioral studies)
- 📈 Marketing (consumer sentiment)
- 🔐 Security (suspicious behavior detection)

This project integrates **deep learning** (CNNs, transfer learning), **emotion psychology**, and **real-time processing** to build an intelligent FED system.

---

## 🎯 Objectives

### 🔹 Primary Objectives

- Develop a robust CNN-based deep learning model to classify facial emotions accurately.
- Use data augmentation to enhance training data and reduce overfitting.

### 🔹 Secondary Objectives

- Compare CNN variants like **VGG16**, **ResNet**, and **Custom CNN**.
- Optimize performance using hyperparameter tuning.
- Explore hybrid models (e.g., CNN + LSTM) for sequential emotion dynamics.

---

## 📚 Literature Summary

- **Early Methods**: Feature extraction via Haar Cascades.
- **Modern Era**: Deep learning, particularly CNNs, dominate the field.
- **Transfer Learning**: Models like **VGGFace**, **ResNet50** show high accuracy.
- **Accuracy**: Over 90% in lab conditions; performance drops in real-world settings due to:
  - Varying lighting
  - Occlusion
  - Ethnic and demographic variation

---

## ❗ Problem Statement

### Key Challenges

- **Dataset Bias**: Lack of demographic diversity can lead to poor generalization.
- **Subtle Emotions**: Difficulty in classifying mixed or mild emotions.
- **Real-Time Constraints**: Need for efficient models that can operate live.

### Project Aim

> To develop a **real-time, high-accuracy facial emotion detection system** that performs reliably across diverse faces and environments.

---

## 📁 Dataset and Design

### 📦 Dataset: **FER2013**

- ~35,000 grayscale images
- 7 emotion classes: `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`

### 🧬 Model Architecture

- Multiple **Convolutional Layers** for feature extraction
- **Pooling Layers** to reduce dimensionality
- **Dropout** to avoid overfitting
- **Fully Connected Layers** ending in a **Softmax** classifier

---

## 🛠️ Methodology

### 📥 Data Acquisition

- Use **FER2013** (via Kaggle or OpenCV)
- Optionally integrate **CK+** or custom datasets

### 🧼 Preprocessing

- Convert images to grayscale
- Resize to **48x48**
- Normalize pixel values to `[0, 1]`
- One-hot encode emotion labels

### 🔄 Data Augmentation

- **Rotation**
- **Zoom**
- **Width/Height shift**
- **Horizontal flip**

Improves robustness to real-world variations.

### 🏗️ Model Architecture

```text
Conv2D (32 filters) → ReLU → MaxPool
Conv2D (64 filters) → ReLU → MaxPool
Conv2D (128 filters) → ReLU → MaxPool
Flatten → Dense(128) → Dropout → Dense(7) → Softmax
