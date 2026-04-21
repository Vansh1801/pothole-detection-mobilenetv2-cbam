#  Pothole Detection Android App

## Overview
This project focuses on detecting potholes from road images using deep learning
and deploying the trained model on an Android device for real-time inference.
The goal is to enable automatic pothole detection with GPS-based location logging
using a smartphone camera.

## Dataset
- Dataset contains 38,385 road images
- Includes pothole and non-pothole road surface images
- **Source:** RDD2022 (Road Damage Dataset 2022)
- **Validation Set:** 14,880 images (balanced)
- **Target:** Binary Classification — Pothole / No Pothole

## Approach

### Data Preprocessing
- Applied data augmentation (flip, rotation, zoom, brightness)
- Handled class imbalance via class balancing techniques
- Resized all images to 224×224 for MobileNetV2 input

### Exploratory Data Analysis (EDA)
- Analyzed class distribution and imbalance
- Visualized sample road images across damage categories

## Model Architecture
- **Backbone:** MobileNetV2 (pretrained on ImageNet)
- **Attention Module:** CBAM (Convolutional Block Attention Module)
  — adds channel and spatial attention to focus on road defect regions
- **Training Strategy:** Two-Phase
  - Phase 1 — Transfer Learning: Froze base layers, trained custom head
  - Phase 2 — Fine-tuning: Unfroze top layers, continued at lower learning rate

## Model Training
**Callbacks Used:**
- `ModelCheckpoint` — saved best weights based on val_loss
- `EarlyStopping` — stopped training when val_loss stopped improving
- `ReduceLROnPlateau` — reduced learning rate on plateau

## Model Evaluation

| Metric | Score |
|--------|-------|
| **Accuracy** | 82.43% |
| **F1-Score** | 84.48% |
| **AUC-ROC** | 91.34% |
| **Recall** | 95.63% |

**Metrics Used:**
Accuracy, F1-Score, AUC-ROC, Recall, Precision, Confusion Matrix, ROC Curve

## Android Deployment
- Converted trained Keras model to **TFLite format (2.55 MB)**
- Integrated TFLite model into Android app built in **Android Studio**
- **On-device inference** — no internet connection required
- **GPS Tagging** — automatically logs pothole location using device GPS

## Key Insights
- High Recall (95.63%) ensures very few potholes are missed — critical for
  real-world road safety applications
- CBAM attention improved the model's focus on pothole regions vs background
- Two-phase training gave better generalization than single-phase fine-tuning
- TFLite conversion reduced model to 2.55 MB without significant accuracy loss

## Tools & Technologies
- Python
- TensorFlow, Keras, TFLite
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn
- Android Studio
- Google Colab (T4 GPU)

## Author
**Vansh Bhatia**
B.E. Computer Science | MMIT, Pune
vanshbhatia1805@gmail.com
