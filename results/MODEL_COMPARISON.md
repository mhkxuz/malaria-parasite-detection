# Model Comparison — Malaria Cell Classification

## Experimental Setup
- Dataset: NIH Malaria Cell Images
- Total images: 27,558
- Training images: 16,544
- Validation images: 5,517
- Test images: 5,497
- Split strategy: Patient-independent
- Image resolution: 128x128 RGB
- Batch size: 32
- Training epochs: 4 per model

## Test Set Performance

| Metric | Custom CNN | MobileNetV2 |
|---|---:|---:|
| Accuracy | 95.23% | 92.01% |
| Precision | 96.73% | 93.30% |
| Recall | 93.64% | 90.55% |
| F1-score | 95.16% | 91.90% |
| ROC-AUC | 98.74% | 97.37% |

## Confusion Matrices

### Custom CNN
| Actual / Predicted | Uninfected | Parasitized |
|---|---:|---:|
| Uninfected | 2658 | 87 |
| Parasitized | 175 | 2577 |

### MobileNetV2
| Actual / Predicted | Uninfected | Parasitized |
|---|---:|---:|
| Uninfected | 2566 | 179 |
| Parasitized | 260 | 2492 |

## Findings

The custom CNN achieved higher performance than MobileNetV2
across all five reported test metrics.

It achieved 95.23% test accuracy and 98.74% ROC-AUC,
compared with 92.01% accuracy and 97.37% ROC-AUC
for MobileNetV2.

The custom CNN also produced fewer false negatives:
175 compared with 260 for MobileNetV2.

## Conclusion

Under the experimental configuration used in this project,
the custom CNN performed better than the MobileNetV2
transfer learning model on the patient-independent test set.

This finding is specific to these models, training settings,
dataset and evaluation protocol.
