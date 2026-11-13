
# Research Methodology

## 1. Research Design

This project follows an experimental comparative research design to investigate deep learning approaches for automated malaria parasite detection.

Two convolutional neural network architectures will be developed and evaluated:

1. Custom Convolutional Neural Network (CNN).
2. MobileNetV2 using transfer learning.

Both models will perform binary classification of microscopic red blood cell images into parasitized and uninfected categories.

The experiments will use patient-independent data partitions to evaluate generalization to previously unseen patients.

## 2. Dataset Acquisition

### 2.1 Dataset Source

The project will use the NIH Malaria Cell Images Dataset published by the National Library of Medicine.

Official source:

https://data.lhncbc.nlm.nih.gov/public/Malaria/

Original research paper:

Rajaraman et al. (2018), PeerJ, 6, e4568.

https://doi.org/10.7717/peerj.4568

### 2.2 Dataset Characteristics

The dataset contains 27,558 segmented red blood cell images.

| Class | Number of Images |
|---|---:|
| Parasitized | 13,779 |
| Uninfected | 13,779 |
| Total | 27,558 |

The images represent two categories:

- Parasitized: Red blood cells containing malaria parasites.
- Uninfected: Red blood cells without visible malaria parasites.

### 2.3 Patient Mapping

The original NIH dataset provides patient-to-cell mapping CSV files.

These mappings will be used to associate individual images with their originating patient identifiers.

The mapping files will be examined to establish:

- Available patient identifiers.
- Image-to-patient associations.
- Duplicate mapping records.
- Missing image associations.
- Patient distribution across both classes.

Patient identifiers will be normalized before grouping.

Images whose patient associations cannot be reliably reconstructed will not silently be assigned arbitrary patient identifiers.

If mapping completeness is insufficient, the limitation will be documented before proceeding with patient-independent evaluation.

## 3. Exploratory Data Analysis

Before model development, exploratory data analysis will be conducted.

The following characteristics will be investigated:

1. Total number of images.
2. Distribution of images across classes.
3. Image dimensions and color channels.
4. Pixel intensity distribution.
5. Representative images from both classes.
6. Number of unique patients.
7. Number of images associated with each patient.
8. Patient-level class distribution.

Visualizations will be generated using Matplotlib.

The analysis will help identify inconsistencies and inform preprocessing decisions.

## 4. Patient-Independent Data Partitioning

### 4.1 Motivation

Random image-level splitting can introduce data leakage when images originating from the same patient occur in multiple partitions.

Images from the same patient may share visual characteristics and acquisition conditions.

This can result in overly optimistic evaluation metrics.

To reduce this risk, the project will use patient-level grouping.

### 4.2 Partition Strategy

The target partition proportions are:

| Partition | Target proportion |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

These are approximate image-count targets. Actual proportions may differ because patients, rather than individual images, are assigned to partitions.

### 4.3 Grouping Procedure

1. Associate every image with its verified patient identifier.
2. Group images using patient identifiers.
3. Assign entire patient groups to training, validation or testing.
4. Ensure that no patient appears in more than one partition.
5. Examine class distributions in each partition.
6. Adjust the group assignment if necessary without compromising patient separation.

A group-aware splitting strategy from scikit-learn will be used.

### 4.4 Leakage Verification

The following conditions must hold:

- Training and validation patient sets must be disjoint.
- Training and testing patient sets must be disjoint.
- Validation and testing patient sets must be disjoint.

The implementation must programmatically verify these conditions.

Image paths must also be checked for duplicates across partitions.

If patient-level grouping cannot be established reliably, the experiment must be reported as limited rather than presented as patient-independent evaluation.

## 5. Image Preprocessing

All images will undergo a consistent preprocessing pipeline.

### 5.1 Image Resizing

Images will be resized to 224 × 224 pixels to support the MobileNetV2 architecture.

### 5.2 Normalization

Pixel values will be converted to floating-point representations.

The custom CNN will use pixel normalization to the [0, 1] range.

MobileNetV2 will use its corresponding preprocessing function, which scales pixel values to the range [-1, 1].

### 5.3 Data Augmentation

Training images may undergo:

- Random horizontal flipping.
- Random vertical flipping.
- Small rotations.
- Limited zoom transformations.

Augmentation will be applied only to training data.

Validation and testing images will retain deterministic preprocessing.

No test images will be used for augmentation, training or hyperparameter selection.

## 6. Model Architecture 1: Custom CNN

A custom Convolutional Neural Network will be designed and implemented using TensorFlow/Keras.

The architecture will contain:

1. Input layer.
2. Convolutional layers.
3. ReLU activation functions.
4. Max-pooling layers.
5. Batch normalization where appropriate.
6. Dropout for regularization.
7. Flattening or global average pooling.
8. Fully connected classification layers.
9. Sigmoid output layer.

Binary cross-entropy will be used as the loss function.

The Adam optimizer will be used for training.

The model will learn visual features directly from the training dataset without pretrained image-classification weights.

## 7. Model Architecture 2: MobileNetV2

MobileNetV2 will be implemented using TensorFlow/Keras and ImageNet pretrained weights.

MobileNetV2 uses inverted residual blocks and depthwise separable convolutions to reduce computational complexity.

### 7.1 Transfer Learning

The pretrained convolutional base will initially be frozen.

A new binary classification head will be attached to the feature extractor.

### 7.2 Fine-Tuning

Where computationally feasible, selected upper layers will subsequently be unfrozen and fine-tuned using a lower learning rate.

Batch normalization layers will be handled carefully during fine-tuning.

### 7.3 Classification Head

The classification head will include:

- Global average pooling.
- Dropout.
- Dense output layer.
- Sigmoid activation.

The model will use binary cross-entropy and the Adam optimizer.

## 8. Training Configuration

Both architectures will use a reproducible training configuration.

Initial experimental settings:

| Parameter | Configuration |
|---|---|
| Input resolution | 224 × 224 |
| Batch size | 16 or 32, depending on memory |
| Optimizer | Adam |
| Loss | Binary cross-entropy |
| Epochs | Up to 15–20 |
| Random seed | 42 |
| Early stopping | Enabled |
| Model checkpointing | Enabled |

The actual number of completed epochs and training configurations will be documented.

Early stopping will monitor validation performance to reduce unnecessary training.

The best validation checkpoint will be retained.

Class weights will be considered if the resulting partitions show meaningful class imbalance.

The test partition will not influence training decisions.

## 9. Model Evaluation

Both models will be evaluated on the same held-out test partition.

The following metrics will be calculated.

### 9.1 Accuracy

The proportion of correctly classified images.

### 9.2 Precision

The proportion of predicted parasitized images that are actually parasitized.

### 9.3 Recall

The proportion of actual parasitized images correctly identified.

### 9.4 F1-Score

The harmonic mean of precision and recall.

### 9.5 Confusion Matrix

The confusion matrix will summarize:

- True Positives.
- True Negatives.
- False Positives.
- False Negatives.

Evaluation will use the same classification threshold for both models unless a different threshold is explicitly justified and selected using validation data.

Results will be presented in tables and graphical comparisons.

## 10. Comparative Analysis

The custom CNN and MobileNetV2 will be compared using:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Training time.
- Model size.
- Inference time, where measurable.

The analysis will investigate the relative advantages of training a custom architecture and using pretrained image features.

No model will be declared superior without supporting experimental results.

## 11. Explainable AI Using Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) will be implemented to visualize image regions influencing model predictions.

Grad-CAM will generate heatmaps using gradients flowing into selected convolutional feature maps.

The visualizations will help investigate whether models focus on relevant microscopic structures when classifying blood cell images.

Grad-CAM will be applied to representative correctly and incorrectly classified examples.

These visualizations will be treated as qualitative explanations rather than proof of clinical reasoning.

Reference:

Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.

https://arxiv.org/abs/1610.02391

## 12. Streamlit Application

A demonstration application will be developed using Streamlit.

The application will support:

1. Uploading a microscopic blood cell image.
2. Displaying the uploaded image.
3. Running the selected trained model.
4. Displaying the predicted class.
5. Displaying prediction confidence.
6. Optionally displaying a Grad-CAM visualization.

The interface will clearly state that the system is an academic prototype and is not intended for clinical diagnosis.

## 13. Reproducibility

To improve reproducibility:

- Random seeds will be configured.
- Dataset partitions will be saved.
- Model configurations will be documented.
- Training histories will be retained.
- Trained model checkpoints will be saved.
- Evaluation outputs will be recorded.
- Software versions will be documented.

Git will be used for version control.

## 14. Limitations

The project has several limitations:

1. The dataset contains segmented cell images rather than complete blood smear images.
2. Dataset-specific performance does not guarantee generalization to other laboratories.
3. Patient-independent evaluation depends on the completeness and correctness of the NIH mapping metadata.
4. Limited computational resources may restrict extensive hyperparameter optimization.
5. Grad-CAM visualizations do not establish clinical validity.
6. Cell-level predictions cannot independently establish whether a patient has malaria.

The system is intended for academic experimentation and demonstration, not medical diagnosis.

## 15. Expected Research Contribution

The project will provide an experimental comparison of a custom CNN and a transfer-learning architecture for malaria parasite detection.

Its primary methodological emphasis will be patient-independent evaluation, reproducibility and explainability.

The study will document the strengths and limitations of both approaches using experimentally obtained results.

