# 🐶🐱 Oxford-IIIT Pets Multi-Task Learning (Classification + Segmentation)

## 📘 Project Overview

This project explores multi-task learning using the Oxford-IIIT Pets Dataset, aiming to simultaneously perform **image classification** and **semantic segmentation** of cat and dog breeds. Two neural network architectures are evaluated:

- A **custom-built Deep CNN** trained from scratch
- A **fine-tuned MobileNetV3Small** leveraging transfer learning

The models are evaluated across various performance metrics, including classification accuracy and segmentation F1 score.

---

## 🧠 Objectives

- Load and preprocess Oxford-IIIT Pets dataset with image-label-mask triplets
- Implement task-specific data augmentation
- Train two multi-output models (classification + segmentation)
- Evaluate and visualize predictions (including side-by-side segmentation mask comparison)
- Compare custom CNN and fine-tuned MobileNetV3Small on performance and training behavior

---

## 🛠️ Tools and Frameworks

- **TensorFlow & Keras** – Deep learning
- **TensorFlow Datasets** – Oxford-IIIT Pet dataset
- **OpenCV** – Image processing (mask prep and grayscale conversion)
- **Matplotlib & Seaborn** – Visualizations
- **Scikit-learn** – Evaluation metrics

---

## 🧪 Key Features

### 📂 Data Processing
- Custom `tf.data.Dataset` loader with:
  - Image-label-mask unpacking
  - Mask preprocessing
  - Augmentations (e.g., random horizontal flip)
  - MobileNet-style normalization
  - Task-based output selector (classification only / segmentation only / both)

### 🧱 Model Architectures
- **DCNN from Scratch**:
  - 4-layer convolutional encoder
  - Dual heads:
    - Classification (Dense layers)
    - Segmentation (Conv2DTranspose decoder)
- **Fine-Tuned MobileNetV3Small**:
  - Pre-trained backbone (frozen then partially unfrozen)
  - Same dual-headed structure as above

### 📊 Evaluation
- **Classification**:
  - Confusion Matrix
  - Accuracy
  - Weighted F1 Score
  - Per-class precision/recall

- **Segmentation**:
  - Pixel-wise Confusion Matrix
  - Binary F1 Score
  - Precision / Recall / Accuracy

- **Visualization**:
  - Dynamic segmentation predictions vs. ground truth
  - Loss curve plots for both tasks
  - Training vs. validation metrics over time

---

## 📈 Example Results

| Model                      | Classification Accuracy | Segmentation F1 Score |
|---------------------------|--------------------------|------------------------|
| Custom DCNN (from scratch)|         ~85%             |        ~0.78           |
| MobileNetV3 (fine-tuned)  |         ~90%             |        ~0.84           |

> *Metrics may vary slightly depending on augmentation, learning rate, and training time.*

---

## 🖼️ Sample Output

| Input Image | Ground Truth Mask | Predicted Mask |
|-------------|-------------------|----------------|
| ![img1](./samples/img1.png) | ![gt1](./samples/gt1.png) | ![pred1](./samples/pred1.png) |
| ![img2](./samples/img2.png) | ![gt2](./samples/gt2.png) | ![pred2](./samples/pred2.png) |

---

⚖️ Ethical Considerations
Dataset is publicly available and non-sensitive

Segmentation masks are used for educational purposes only

No identifiable human data is involved

👨‍💻 Author
Mohan Hao
Machine Learning Enthusiast
📧 imhaom@gmail.com
