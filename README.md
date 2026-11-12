# Crop Disease Detection Using Deep Learning

Beginner-friendly project aligned with **SDG 2: Zero Hunger**. It classifies crop-leaf images using a MobileNetV2 transfer-learning model.

## Status
Initial scaffold only: code is provided, but the dataset has not been downloaded and the model has not been trained.

## Environment
Use Python 3.11 in a virtual environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Python 3.11 is unavailable on Arch, create a Python 3.11 Conda environment.

## Dataset
Download PlantVillage from a reputable source and check its license/terms. Arrange it as one directory per class:

```text
data/raw/PlantVillage/
├── Tomato___Early_blight/
├── Tomato___healthy/
└── ...
```

Then split it:

```bash
python -m src.split_dataset --input data/raw/PlantVillage --output data/processed
```

## Train, evaluate, predict, run app

```bash
python -m src.train --data data/processed --epochs 8 --batch-size 32
python -m src.evaluate --data data/processed
python -m src.predict path/to/leaf.jpg
streamlit run app/streamlit_app.py
```

The initial model freezes ImageNet-pretrained MobileNetV2 and trains a new classification head. This is transfer learning.

## Important limitations
- Strong results on PlantVillage do not prove real-world field performance; images may have controlled backgrounds and lighting.
- The model only predicts among classes it has seen.
- Confidence scores are not calibrated probabilities.
- This is an educational screening prototype, not agricultural advice.

## Suggested team division
1. Dataset research, attribution, preprocessing.
2. Model training and architecture documentation.
3. Evaluation, error analysis, visualizations.
4. Streamlit app, integration, report/presentation (if four members).

Everyone should understand the full pipeline and be able to explain it.
