# Research Methodology

## 1. Research Design

This project followed an experimental comparative research design to investigate deep learning approaches for automated malaria parasite detection from microscopic red blood cell images.

Two convolutional neural network architectures were implemented and evaluated:

1. Custom Convolutional Neural Network (CNN).
2. MobileNetV2 using transfer learning.

Both models performed binary classification of segmented red blood cell images into two categories: Parasitized and Uninfected.

A patient-independent evaluation strategy was adopted to assess model performance on images originating from patients not represented in the corresponding training or validation partitions.

The study also incorporated Gradient-weighted Class Activation Mapping (Grad-CAM) to qualitatively examine model activation patterns.

## 2. Dataset Acquisition

### 2.1 Dataset Source

The project used the NIH Malaria Cell Images Dataset published by the National Library of Medicine.

Official source:

https://data.lhncbc.nlm.nih.gov/public/Malaria/

Original research paper:

Rajaraman et al. (2018), PeerJ, 6, e4568.

https://doi.org/10.7717/peerj.4568

### 2.2 Dataset Characteristics

The dataset contains 27,558 segmented red blood cell images belonging to two equally represented classes.

| Class | Number of Images |
|---|---:|
| Parasitized | 13,779 |
| Uninfected | 13,779 |
| **Total** | **27,558** |

The two categories are:

- **Parasitized:** Red blood cell images containing malaria parasites.
- **Uninfected:** Red blood cell images without visible malaria parasites.

### 2.3 Patient Mapping

The original NIH dataset provides patient-to-cell mapping CSV files for both image categories.

These mapping files were used to associate each image with its originating patient identifier.

The mapping audit established:

- 151 patient identifiers in the Parasitized mapping.
- 201 patient identifiers in the Uninfected mapping.
- 151 patient identifiers shared between the two categories.
- 201 unique patient identifiers across the complete dataset.

All 27,558 images were successfully associated with patient identifiers. The mapping validation also checked for duplicate records, missing image associations and unmapped images.

Patient identifiers were used as grouping variables during data partitioning to prevent images from the same patient from appearing in multiple partitions.

## 3. Exploratory Data Analysis

Exploratory data analysis was conducted before model development.

The analysis examined:

1. Total number of images.
2. Distribution of images across the two classes.
3. Image dimensions and color channels.
4. Representative images from both classes.
5. Number of unique patients.
6. Number of images associated with each patient.
7. Patient-level class distribution.

Matplotlib was used to generate visualizations and inspect representative samples.

The analysis confirmed that the dataset contained an equal number of images in each class and that patient mapping metadata was available for patient-independent partitioning.

## 4. Patient-Independent Data Partitioning

### 4.1 Motivation

Random image-level splitting can introduce data leakage when images originating from the same patient appear in multiple partitions.

Images from the same patient may share visual characteristics and acquisition conditions. Consequently, an image-level split can produce overly optimistic evaluation results.

To reduce this risk, the project used patient-level grouping during dataset partitioning.

### 4.2 Partition Strategy

The dataset was divided into training, validation and testing partitions using an approximate 60/20/20 image distribution.

| Partition | Images | Patients | Approximate Proportion |
|---|---:|---:|---:|
| Training | 16,544 | 120 | 60% |
| Validation | 5,517 | 41 | 20% |
| Testing | 5,497 | 40 | 20% |
| **Total** | **27,558** | **201** | **100%** |

The actual proportions were determined by assigning complete patient groups rather than individual images.

### 4.3 Grouping Procedure

The partitioning procedure was implemented using `StratifiedGroupKFold` from scikit-learn.

The following procedure was used:

1. Every image was associated with its verified patient identifier.
2. Patient identifiers were used as group labels.
3. An outer `StratifiedGroupKFold` split with five folds, shuffling enabled and random seed 42 was used to reserve the test partition.
4. The remaining development data was divided into training and validation partitions using a second `StratifiedGroupKFold` split with four folds, shuffling enabled and random seed 42.
5. Class distributions were examined across the resulting partitions.

Stratification was used to maintain reasonably balanced class distributions while preserving patient-level separation.

### 4.4 Leakage Verification

The implementation programmatically verified that:

- Training and validation patient sets were disjoint.
- Training and testing patient sets were disjoint.
- Validation and testing patient sets were disjoint.

The resulting partitions contained 120 training patients, 41 validation patients and 40 testing patients, with no patient overlap.

The same saved partition assignments were used for both model experiments to ensure a consistent comparison.

## 5. Image Preprocessing

All images were processed using a consistent input pipeline implemented with TensorFlow.

### 5.1 Image Resizing

All images were resized to **128 × 128 pixels** with three RGB color channels.

The resulting input shape for both architectures was:

`128 × 128 × 3`

### 5.2 Normalization

Images were decoded as RGB and converted to floating-point representations.

The normalization strategy depended on the model architecture:

- **Custom CNN:** Pixel values were rescaled from the original range [0, 255] to [0, 1] using a Keras Rescaling layer with a scale factor of 1/255.
- **MobileNetV2:** Pixel values were rescaled from [0, 255] to [-1, 1] using a scale factor of 1/127.5 and an offset of -1.

### 5.3 Data Pipeline

The TensorFlow data pipeline performed the following operations:

1. Read image paths and labels from the saved dataset partitions.
2. Decode PNG images into RGB format.
3. Resize images to 128 × 128 pixels.
4. Convert image data to `float32`.
5. Apply training-data shuffling.
6. Batch images using a batch size of 32.
7. Prefetch batches using TensorFlow's `AUTOTUNE` mechanism.

No data augmentation was applied during the reported experiments. Validation and testing images underwent deterministic preprocessing.

The same image partitions and input resolution were used for both models.

## 6. Model Architecture 1: Custom CNN

A custom Convolutional Neural Network was designed and implemented using TensorFlow/Keras.

The model learned image features directly from the training dataset without pretrained image-classification weights.

### 6.1 Architecture

The architecture consisted of the following components:

1. Input layer accepting images of shape 128 × 128 × 3.
2. Rescaling layer to normalize pixel values to [0, 1].
3. Four convolutional blocks containing 16, 32, 64 and 128 filters respectively.
4. Each convolutional layer used a 3 × 3 kernel with same padding and no bias term.
5. Batch normalization and ReLU activation followed each convolutional layer.
6. Max-pooling was applied after the first three convolutional blocks.
7. Global Average Pooling reduced the final feature maps.
8. A Dropout layer with a rate of 0.4 was applied.
9. A fully connected Dense layer containing 64 neurons with ReLU activation.
10. A second Dropout layer with a rate of 0.3.
11. A final Dense layer with one neuron and sigmoid activation for binary classification.

The model used binary cross-entropy as its loss function and Adam as its optimizer.

The initial learning rate was set to 0.001.

## 7. Model Architecture 2: MobileNetV2

MobileNetV2 was implemented using TensorFlow/Keras with ImageNet-pretrained weights.

MobileNetV2 uses inverted residual blocks and depthwise separable convolutions to reduce computational complexity.

### 7.1 Transfer Learning

The MobileNetV2 architecture was initialized using ImageNet-pretrained weights with `include_top=False`.

The pretrained convolutional base was frozen during training. Its weights were not updated.

The base model was called with `training=False` to maintain inference behavior within the frozen feature extractor.

### 7.2 Fine-Tuning

Fine-tuning was not performed in the reported experiment.

The pretrained convolutional base remained frozen throughout training. The experiment therefore evaluated feature extraction using pretrained ImageNet representations followed by a newly trained classification head.

### 7.3 Classification Head

The classification head consisted of:

- Global Average Pooling.
- Dropout with a rate of 0.3.
- Dense output layer with one neuron.
- Sigmoid activation.

The model used binary cross-entropy and the Adam optimizer with an initial learning rate of 0.001.

## 8. Training Configuration

Both architectures were trained using the same dataset partitions and a consistent experimental setup.

| Parameter | Custom CNN | MobileNetV2 |
|---|---|---|
| Input resolution | 128 × 128 | 128 × 128 |
| Batch size | 32 | 32 |
| Optimizer | Adam | Adam |
| Initial learning rate | 0.001 | 0.001 |
| Loss function | Binary cross-entropy | Binary cross-entropy |
| Maximum epochs | 4 | 4 |
| Random seed for partitioning | 42 | 42 |
| Early stopping | Enabled | Enabled |
| Model checkpointing | Enabled | Enabled |
| Learning-rate reduction | Enabled | Enabled |
| Pretrained weights | No | ImageNet |
| Fine-tuning | Not applicable | Not performed |

The training process used the following callbacks:

- **ModelCheckpoint:** Saved the best-performing model checkpoint based on validation loss.
- **EarlyStopping:** Monitored validation loss and restored the best model weights when training stopped.
- **ReduceLROnPlateau:** Reduced the learning rate by a factor of 0.5 when validation loss stopped improving, subject to a minimum learning rate of 0.000001.

The Custom CNN used an early-stopping patience of 3 epochs and a learning-rate reduction patience of 2 epochs.

MobileNetV2 used an early-stopping patience of 2 epochs and a learning-rate reduction patience of 1 epoch.

Both models completed all four training epochs. The learning rate remained at 0.001 throughout both runs, and no learning-rate reduction was triggered.

Training histories were saved as JSON files for subsequent analysis and visualization.

## 9. Model Evaluation

Both models were evaluated using the same held-out test partition containing 5,497 images from 40 previously unseen patients.

The following evaluation metrics were calculated.

### 9.1 Accuracy

The proportion of correctly classified images among all evaluated images.

### 9.2 Precision

The proportion of images predicted as Parasitized that were actually Parasitized.

### 9.3 Recall

The proportion of actual Parasitized images correctly identified by the model.

### 9.4 F1-Score

The harmonic mean of precision and recall, providing a combined measure of both metrics.

### 9.5 ROC-AUC

The Area Under the Receiver Operating Characteristic Curve measures the model's ability to distinguish between the two classes across different classification thresholds.

### 9.6 Confusion Matrix

The confusion matrix was used to examine classification outcomes:

- True Positives.
- True Negatives.
- False Positives.
- False Negatives.

Both models used a sigmoid classification threshold of 0.5 for the reported binary predictions.

Evaluation results were saved as JSON files and visualized using confusion matrices and comparative performance plots.

## 10. Comparative Analysis

The Custom CNN and MobileNetV2 were compared using the following metrics:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- Confusion matrix outcomes.

Training histories were also examined to compare changes in training and validation accuracy and loss across epochs.

The analysis investigated the relative performance of a custom architecture trained from scratch and a transfer-learning architecture using frozen ImageNet-pretrained features.

Both models were evaluated on identical test data, allowing their results to be compared under the same patient-independent evaluation conditions.

Conclusions were restricted to the experimental configuration and dataset used in this project.

## 11. Explainable AI Using Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) was implemented to visualize image regions associated with model predictions.

Grad-CAM generates heatmaps using gradients flowing into selected convolutional feature maps.

For the Custom CNN, the final convolutional layer was used to generate Grad-CAM visualizations.

The resulting feature map had dimensions of 16 × 16 × 128 and was upsampled to 128 × 128 pixels for visualization.

Grad-CAM was applied to representative examples, including:

- True Positive.
- True Negative.
- False Positive.
- False Negative.

The visualizations were used to qualitatively examine model activation patterns and investigate how the network responded to different image regions.

Grad-CAM was treated as an interpretation aid rather than proof that the model identified biologically meaningful parasite structures. In particular, activation patterns near image boundaries or corners were considered potential indications of sensitivity to image edges or background characteristics.

These visualizations do not establish clinical reasoning, parasite localization accuracy or diagnostic validity.

Reference:

Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.

https://arxiv.org/abs/1610.02391

## 12. Streamlit Application

A demonstration application was developed using Streamlit to provide an interactive interface for the trained Custom CNN.

The application supports:

1. Uploading a microscopic red blood cell image.
2. Displaying the uploaded image.
3. Running the trained classification model.
4. Displaying the model's predicted class.
5. Displaying the model's prediction score.
6. Generating Grad-CAM visualizations.
7. Viewing model performance metrics and experimental results.
8. Exploring representative examples from the dataset.

The application presents the model prediction as an experimental output rather than a medical diagnosis.

It explicitly states that the system is an academic prototype and is not intended for clinical use.

## 13. Reproducibility

Several measures were adopted to improve experimental reproducibility:

- Patient-to-image mapping metadata was retained.
- Dataset partitions were generated using a fixed random seed.
- The resulting dataset assignments were saved.
- Both models used the same training, validation and testing partitions.
- Model architectures and training configurations were implemented in separate Python modules.
- Training histories were retained as JSON files.
- Trained model checkpoints were saved.
- Evaluation metrics and prediction outputs were recorded.
- Confusion matrices, learning curves and Grad-CAM visualizations were generated and retained.
- Git was used for source-code version control.

The project was implemented using Python and TensorFlow/Keras, with scikit-learn used for patient-independent data partitioning and evaluation support.

## 14. Limitations

The project has several limitations:

1. The dataset contains segmented cell images rather than complete blood smear images.
2. The evaluation was performed using a single publicly available dataset.
3. Patient-independent performance within the NIH dataset does not establish generalization to other laboratories, imaging systems, staining procedures or patient populations.
4. MobileNetV2 was evaluated using frozen pretrained features; fine-tuning was not performed.
5. No data augmentation or extensive hyperparameter optimization was performed in the reported experiments.
6. The computational resources available limited the scope of experimentation.
7. Grad-CAM visualizations do not establish that the model relies on clinically meaningful features.
8. Cell-level predictions cannot independently establish whether a patient has malaria.

The test partition contained previously unseen patients from the same NIH dataset. It did not constitute external clinical validation.

The system is intended for academic experimentation and demonstration, not medical diagnosis.

## 15. Research Contribution

This project provides an experimental comparison between a custom CNN trained from scratch and MobileNetV2 using frozen ImageNet-pretrained features for malaria parasite classification.

Its primary methodological emphasis is patient-independent evaluation through patient-grouped stratified partitioning.

The study also combines quantitative model evaluation with qualitative Grad-CAM visualizations and an interactive Streamlit demonstration application.

The resulting experiments document the performance and limitations of both architectures under the same dataset and evaluation conditions.