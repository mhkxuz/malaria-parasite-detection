# Project plan and report checklist

## Problem statement
Crop diseases can reduce agricultural productivity. This project explores whether a
deep-learning image classifier can identify selected crop-leaf disease categories from
photographs and provide an accessible screening prototype.

## Objectives
- Prepare and document a labelled crop-leaf image dataset.
- Train a MobileNetV2 transfer-learning classifier using TensorFlow/Keras.
- Evaluate performance on a held-out test set.
- Build an image-upload interface.
- Discuss limitations, dataset bias, and responsible use.

## Report evidence
- Dataset source, license/terms, class list, image counts, and attribution.
- Train/validation/test split and random seed.
- Preprocessing and augmentation choices.
- Architecture diagram and transfer-learning explanation.
- Optimizer, learning rate, batch size, epochs, and loss.
- Training/validation curves.
- Test accuracy, per-class precision/recall/F1, confusion matrix.
- Error analysis and limitations.

## Questions to settle
1. Which exact PlantVillage source/release was used?
2. How many images and classes were retained?
3. Were duplicates or near-duplicates checked across splits?
4. Does performance change on field photographs?
5. What are the consequences of a wrong prediction?
