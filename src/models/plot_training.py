from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

MODELS = {
    "Custom CNN": "custom_cnn_history.json",
    "MobileNetV2": "mobilenetv2_history.json",
}


def load_history(filename):
    path = RESULTS / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


histories = {
    name: load_history(filename)
    for name, filename in MODELS.items()
}

# Individual accuracy and loss curves for each model.
for name, history in histories.items():
    epochs = range(1, len(history["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["accuracy"], marker="o", label="Training")
    axes[0].plot(epochs, history["val_accuracy"], marker="o", label="Validation")
    axes[0].set_title(f"{name} - Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(list(epochs))
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["loss"], marker="o", label="Training")
    axes[1].plot(epochs, history["val_loss"], marker="o", label="Validation")
    axes[1].set_title(f"{name} - Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Binary cross-entropy")
    axes[1].set_xticks(list(epochs))
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"{name} Training History")
    fig.tight_layout()
    output = RESULTS / f"{name.lower().replace(' ', '_')}_learning_curves.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", output)

# Direct comparison of validation accuracy and loss.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for name, history in histories.items():
    epochs = range(1, len(history["val_accuracy"]) + 1)
    axes[0].plot(epochs, history["val_accuracy"], marker="o", label=name)
    axes[1].plot(epochs, history["val_loss"], marker="o", label=name)

axes[0].set_title("Validation Accuracy Comparison")
axes[0].set_ylabel("Accuracy")
axes[1].set_title("Validation Loss Comparison")
axes[1].set_ylabel("Binary cross-entropy")

for axis in axes:
    axis.set_xlabel("Epoch")
    axis.legend()
    axis.grid(alpha=0.3)

fig.tight_layout()
output = RESULTS / "model_validation_comparison.png"
fig.savefig(output, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", output)
print("\nLearning curve generation completed.")
