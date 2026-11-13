
# Patient-Independent Malaria Parasite Detection Using Deep Learning

## 1. Abstract

Malaria remains a significant global health challenge, particularly in regions where access to reliable diagnostic facilities is limited. Microscopic examination of Giemsa-stained blood smears is a widely used method for detecting malaria parasites, but manual examination can be time-consuming and requires trained personnel.

This project investigates deep learning techniques for automated classification of microscopic red blood cell images into two categories: parasitized and uninfected.

Two convolutional neural network approaches will be implemented and compared: a custom CNN developed from scratch and MobileNetV2 using transfer learning.

The models will be evaluated using patient-independent data partitions to reduce information leakage between training and testing datasets. Performance will be measured using accuracy, precision, recall, F1-score and confusion matrices.

Grad-CAM will additionally be used to visualize image regions influencing model predictions.

## 2. Introduction

Deep learning has demonstrated considerable potential in medical image analysis, particularly in image classification and pattern recognition.

Convolutional Neural Networks (CNNs) are capable of learning spatial features directly from images, making them suitable for analyzing microscopic blood cell images.

This project focuses on binary classification of red blood cell images obtained from the NIH Malaria Cell Images dataset.

## 3. Problem Statement

Traditional microscopic malaria examination requires trained personnel to identify infected cells.

The objective is to investigate whether deep learning models can automatically distinguish parasitized red blood cells from uninfected cells.

A major methodological concern is data leakage. Images originating from the same patient may share visual characteristics. Random image-level splitting can therefore produce overly optimistic evaluation results.

This project will investigate patient-independent evaluation using the patient-to-cell mapping files provided by the NIH.

## 4. Research Question

How effectively can custom CNN and MobileNetV2 architectures classify parasitized and uninfected red blood cell images when evaluated on patients excluded from model training?

## 5. Research Objectives

1. Acquire and analyze the NIH Malaria Cell Images dataset.
2. Reconstruct patient-level image groupings using official NIH mapping files.
3. Develop patient-independent training, validation and testing partitions.
4. Design and train a custom CNN architecture.
5. Implement MobileNetV2 transfer learning.
6. Compare both architectures using standard classification metrics.
7. Apply Grad-CAM to investigate model predictions.
8. Develop a Streamlit application for demonstrating image classification.

## 6. SDG Alignment

**United Nations Sustainable Development Goal 3: Good Health and Well-being.**

The project explores computational approaches that may support malaria screening research and automated microscopic image analysis.

It is an academic research prototype and is not intended to replace professional medical diagnosis.

## 7. Dataset

**Dataset:** NIH Malaria Cell Images

**Source:** National Library of Medicine, National Institutes of Health.

**Total images:** 27,558

- Parasitized: 13,779
- Uninfected: 13,779

The dataset consists of segmented red blood cell images obtained from thin blood smear slides.

Official patient-to-cell mapping CSV files will be used to reconstruct patient-level groups.

Dataset source:
https://data.lhncbc.nlm.nih.gov/public/Malaria/

Original research:
https://doi.org/10.7717/peerj.4568

## 8. Proposed Methodology

### 8.1 Dataset acquisition

Download the official NIH image archive and patient mapping files.

### 8.2 Exploratory data analysis

Examine:

- Class distribution
- Image dimensions
- Image quality
- Patient distribution
- Patient-level class distribution

### 8.3 Data partitioning

Create training, validation and testing partitions using patient-level grouping.

No patient identifier should occur across multiple partitions.

### 8.4 Preprocessing

- Image resizing
- Pixel normalization
- Training-only data augmentation
- Consistent preprocessing across both architectures

### 8.5 Custom CNN

Develop a convolutional neural network from scratch using TensorFlow/Keras.

### 8.6 Transfer Learning

Implement MobileNetV2 using pretrained ImageNet weights.

Compare frozen-feature extraction and fine-tuning where computationally feasible.

### 8.7 Evaluation

Evaluate using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

### 8.8 Explainability

Use Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize image regions contributing to predictions.

## 9. Expected Outcomes

The project aims to:

1. Develop two functional deep learning classification models.
2. Compare custom CNN and transfer learning approaches.
3. Evaluate performance on previously unseen patient groups.
4. Produce explainability visualizations.
5. Develop an interactive demonstration application.

No particular performance level is assumed in advance.

## 10. Limitations

- The dataset contains segmented cell images rather than complete blood smear images.
- Results may not generalize to different laboratories, microscopes or staining conditions.
- Cell-level classification does not establish patient-level malaria diagnosis.
- Grad-CAM provides visual explanations but does not establish clinical reasoning.
- Performance on this dataset alone cannot establish clinical readiness.

## 11. References

1. Rajaraman, S., Antani, S. K., Poostchi, M., et al. (2018). Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images. PeerJ, 6, e4568. https://doi.org/10.7717/peerj.4568

2. National Library of Medicine. NIH Malaria Cell Images Dataset. https://data.lhncbc.nlm.nih.gov/public/Malaria/

3. TensorFlow Datasets. Malaria Dataset. https://www.tensorflow.org/datasets/catalog/malaria

4. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. https://arxiv.org/abs/1610.02391

