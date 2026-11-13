# Patient-Independent Malaria Parasite Detection Using Deep Learning

A deep learning research project that classifies thin blood-smear cell images as **Parasitized** or **Uninfected** using TensorFlow. The project compares a custom Convolutional Neural Network (CNN) with a transfer-learning model based on MobileNetV2.

> **Research prototype only.** This project is not a medical diagnostic tool and is not intended for clinical use.

## Project Overview

The objective is to investigate whether deep learning models can classify malaria-infected and uninfected cell images while evaluating them on a patient-independent test set.

### Models

- **Custom CNN:** A CNN designed and trained from scratch.
- **MobileNetV2:** An ImageNet-pretrained model used as a frozen feature extractor.

Both models use the same dataset split and image preprocessing dimensions (128 × 128).

## Dataset

- **Dataset:** NIH Malaria Cell Images
- **Total images:** 27,558
- **Classes:** Parasitized (13,779), Uninfected (13,779)
- **Patients:** 201

The dataset is divided using `StratifiedGroupKFold`, grouping images by patient to prevent patient overlap between training, validation, and test sets.

| Split | Images | Patients |
|---|---:|---:|
| Training | 16,544 | 120 |
| Validation | 5,517 | 41 |
| Testing | 5,497 | 40 |

Dataset source: [NIH Malaria Dataset](https://data.lhncbc.nlm.nih.gov/public/Malaria/)

## Test Results

| Metric | Custom CNN | MobileNetV2 |
|---|---:|---:|
| Accuracy | 95.23% | 92.01% |
| Precision | 96.73% | 93.30% |
| Recall | 93.64% | 90.55% |
| F1-score | 95.16% | 91.90% |
| ROC-AUC | 98.74% | 97.37% |

The custom CNN achieved higher scores across all five reported metrics in this experimental setup. These results do not establish that it will outperform MobileNetV2 on other datasets or clinical settings.

## Explainability

Grad-CAM is used to visualize regions of an input image associated with the model's prediction. These visualizations indicate model activation and sensitivity; they do not prove that the model has learned biologically meaningful parasite features.

## Streamlit Application

The interactive research dashboard includes:

- Project overview
- Cell image classification
- Grad-CAM explainability
- Model performance and evaluation metrics
- Research methodology and limitations

### Run locally

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
