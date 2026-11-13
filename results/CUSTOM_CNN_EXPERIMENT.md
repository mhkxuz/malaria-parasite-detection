# Custom CNN Experiment

## Configuration
- Architecture: Custom CNN
- Dataset: NIH Malaria Cell Images
- Total images: 27,558
- Input resolution: 128x128 RGB
- Training images: 16,544
- Validation images: 5,517
- Test images: 5,497
- Split strategy: Patient-independent
- Training epochs: 4
- Optimizer: Adam
- Loss: Binary Crossentropy
- Learning rate: 0.001
- Batch size: 32

## Training Results
- Training accuracy: 95.87%
- Training loss: 0.1318
- Validation accuracy: 93.80%
- Validation loss: 0.1778
- Validation precision: 97.00%
- Validation recall: 90.23%

## Test Results
- Accuracy: 95.23%
- Precision: 96.73%
- Recall: 93.64%
- F1-score: 95.16%
- ROC-AUC: 98.74%

## Confusion Matrix

| Actual / Predicted | Uninfected | Parasitized |
|---|---:|---:|
| Uninfected | 2658 | 87 |
| Parasitized | 175 | 2577 |

## Artifacts
- custom_cnn_best.keras
- custom_cnn_final.keras
- custom_cnn_history.json
- custom_cnn_test_metrics.json
- custom_cnn_test_predictions.csv
- custom_cnn_confusion_matrix.png
