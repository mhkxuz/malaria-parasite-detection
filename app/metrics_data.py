"""
Experimental metrics, split statistics, and metadata for the
Patient-Independent Malaria Parasite Detection project.
"""

from typing import Dict, Any

PROJECT_TITLE = "Patient-Independent Malaria Parasite Detection Using Deep Learning"
DATASET_NAME = "NIH Malaria Cell Images Dataset"
SOURCE_CITATION = "Rajaraman et al. (2018), PeerJ 6:e4568"
SOURCE_URL = "https://data.lhncbc.nlm.nih.gov/public/Malaria/"

# Model Details
MODEL_FILE = "models/custom_cnn_best.keras"
INPUT_SHAPE = (128, 128, 3)
CLASSIFICATION_THRESHOLD = 0.5
CLASS_NAMES = {0: "Uninfected", 1: "Parasitized"}

# Patient-Independent Dataset Split Statistics
DATASET_STATS: Dict[str, Any] = {
    "total_images": 27558,
    "total_patients": 201,
    "input_resolution": "128 × 128 RGB",
    "split_strategy": "Patient-Independent Group Split (StratifiedGroupKFold)",
    "splits": {
        "train": {
            "images": 16544,
            "percentage": "60.0%",
            "patients": 120,
            "uninfected": 8239,
            "parasitized": 8305,
        },
        "validation": {
            "images": 5517,
            "percentage": "20.0%",
            "patients": 41,
            "uninfected": 2795,
            "parasitized": 2722,
        },
        "test": {
            "images": 5497,
            "percentage": "20.0%",
            "patients": 40,
            "uninfected": 2745,
            "parasitized": 2752,
        },
    },
}

# Experimental Results on Patient-Independent Test Set (Custom CNN)
CUSTOM_CNN_METRICS: Dict[str, Any] = {
    "accuracy": 0.9523,
    "precision": 0.9673,
    "recall": 0.9364,
    "f1_score": 0.9516,
    "roc_auc": 0.9874,
    "test_images": 5497,
    "confusion_matrix": {
        "true_negative": 2658,   # Actual Uninfected, Predicted Uninfected
        "false_positive": 87,     # Actual Uninfected, Predicted Parasitized
        "false_negative": 175,    # Actual Parasitized, Predicted Uninfected
        "true_positive": 2577,    # Actual Parasitized, Predicted Parasitized
    },
    "per_class": {
        "Uninfected": {
            "precision": 0.9382,
            "recall": 0.9683,
            "f1_score": 0.9530,
            "support": 2745,
        },
        "Parasitized": {
            "precision": 0.9673,
            "recall": 0.9364,
            "f1_score": 0.9516,
            "support": 2752,
        },
    },
}

# MobileNetV2 Comparative Results
MOBILENETV2_METRICS: Dict[str, Any] = {
    "accuracy": 0.9201,
    "precision": 0.9330,
    "recall": 0.9055,
    "f1_score": 0.9190,
    "roc_auc": 0.9737,
    "test_images": 5497,
    "confusion_matrix": {
        "true_negative": 2566,
        "false_positive": 179,
        "false_negative": 260,
        "true_positive": 2492,
    },
}

DISCLAIMER_TEXT = (
    "Educational & Research Prototype: This application is developed strictly for research, "
    "academic evaluation, and scientific demonstration purposes. It is not an FDA-cleared or CE-marked "
    "clinical diagnostic medical device and must never be used as a substitute for professional medical "
    "diagnosis, laboratory blood smear examination, or physician consultation."
)
