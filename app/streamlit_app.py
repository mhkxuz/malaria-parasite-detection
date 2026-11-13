"""
Patient-Independent Malaria Parasite Detection — Streamlit Research Application.
"""

import os
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Setup headless environments and suppress low-level logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from app.metrics_data import (
    PROJECT_TITLE,
    DATASET_NAME,
    SOURCE_CITATION,
    SOURCE_URL,
    DATASET_STATS,
    CUSTOM_CNN_METRICS,
    MOBILENETV2_METRICS,
    DISCLAIMER_TEXT,
    MODEL_FILE,
)
from app.model_utils import (
    load_trained_model,
    preprocess_image,
    predict_cell,
    find_last_conv_layer,
    get_all_conv_layers,
    TARGET_SIZE,
)
from app.gradcam_utils import (
    compute_gradcam_heatmap,
    overlay_gradcam_on_image,
    explain_gradcam_findings,
)
from app.ui_components import (
    apply_custom_css,
    render_header,
    render_disclaimer,
    render_metric_card,
    render_prediction_badge,
)

RESULTS_DIR = ROOT / "results"
SAMPLES_DIR = ROOT / "app/sample_images"

# Configure Streamlit page
st.set_page_config(
    page_title="Malaria Parasite Detection | Research Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply medical-AI research styling
apply_custom_css()


# ==============================================================================
# Helper Functions
# ==============================================================================
def get_sample_images_dict():
    """Retrieve available sample images for one-click testing."""
    samples = {}
    if SAMPLES_DIR.is_dir():
        for file in sorted(SAMPLES_DIR.glob("*.png")):
            clean_name = (
                file.stem.replace("_", " ")
                .title()
                .replace("Sample", "Sample #")
            )
            samples[clean_name] = file
    return samples


# ==============================================================================
# Sidebar Navigation
# ==============================================================================
with st.sidebar:
    st.markdown("### 🔬 Malaria AI Research")
    st.markdown(
        "<div style='font-size: 0.85rem; color: #64748b; margin-top: -0.5rem; margin-bottom: 1rem;'>"
        "Deep Learning Research Prototype</div>",
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "🔬 Overview",
            "🧪 Cell Classification",
            "🔍 Grad-CAM Explainability",
            "📊 Model Performance",
            "📖 About the Research",
        ],
        index=0,
    )

    st.markdown("---")

    # Quick model status widget
    st.markdown("#### Model Runtime Status")
    try:
        model = load_trained_model()
        last_conv = find_last_conv_layer(model)
        st.success(f"Custom CNN: Online\nLayer: `{last_conv}`")
    except Exception as e:
        st.error(f"Model Error: {e}")

    st.markdown(
        "<div style='font-size: 0.75rem; color: #64748b; margin-top: 1rem;'>"
        "<strong>Evaluation Benchmark:</strong><br>"
        "• Accuracy: 95.23%<br>"
        "• Recall (Sensitivity): 93.64%<br>"
        "• ROC-AUC: 98.74%<br>"
        "• Patient-Independent Split"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("NIH Malaria Dataset • Deep Learning Lab")


# ==============================================================================
# Top Header & Disclaimer
# ==============================================================================
render_header(
    title=PROJECT_TITLE,
    subtitle="Automated identification of Plasmodium-infected erythrocytes evaluated on patient-independent partitions.",
    tag="Research Prototype",
)
render_disclaimer()


# ==============================================================================
# PAGE 1: Overview
# ==============================================================================
if page == "🔬 Overview":
    st.markdown("### Executive Summary")
    st.write(
        """
        Malaria remains one of the world's most severe global health threats, causing over 600,000
        deaths annually. Manual light-microscopy examination of Giemsa-stained thin blood smears remains
        the diagnostic gold standard, but it is time-consuming, labor-intensive, and prone to inter-observer
        variability, especially in resource-limited rural clinics.

        This research project develops an automated research prototype utilizing a
        **Custom Convolutional Neural Network (CNN)**. To reduce patient data leakage and evaluate
        generalization to previously unseen individuals, the project enforces a patient-level
        grouping strategy (`StratifiedGroupKFold`), ensuring that cells from patients in the testing
        partition were withheld from model training.
        """
    )

    st.markdown("### Key Benchmark Results (Test Set)")
    st.info(
        "Note: The metrics below reflect experimental performance evaluated across 5,497 images "
        "from 40 patients partitioned from the NIH dataset. Patient-independent evaluation within "
        "a single dataset evaluates generalization across individuals but does not constitute external "
        "validation across different clinical centers, staining protocols, or microscope hardware."
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_metric_card("Accuracy", f"{CUSTOM_CNN_METRICS['accuracy']*100:.2f}%", "Overall correctness")
    with col2:
        render_metric_card("Precision", f"{CUSTOM_CNN_METRICS['precision']*100:.2f}%", "Positive predictive value")
    with col3:
        render_metric_card("Recall (Sens.)", f"{CUSTOM_CNN_METRICS['recall']*100:.2f}%", "Detection sensitivity")
    with col4:
        render_metric_card("F1-Score", f"{CUSTOM_CNN_METRICS['f1_score']*100:.2f}%", "Harmonic mean")
    with col5:
        render_metric_card("ROC-AUC", f"{CUSTOM_CNN_METRICS['roc_auc']*100:.2f}%", "Discrimination capability")

    st.markdown("---")

    col_arch, col_flow = st.columns([1, 1])

    with col_arch:
        st.markdown("### System Architecture")
        st.markdown(
            """
            - **Input Layer:** 128 × 128 × 3 RGB microscopic thin-blood smear red blood cell images.
            - **Normalization:** Embedded `Rescaling(1/255)` layer ensuring standardized zero-overhead scaling.
            - **Feature Extraction:** 4 Conv2D blocks with Batch Normalization, ReLU activations, and MaxPooling2D.
            - **Regularization:** Global Average Pooling (GAP) and dual Dropout (0.4 and 0.3) to prevent overfitting.
            - **Output Layer:** Single unit with Sigmoid activation producing model probability for parasitized status.
            - **Explainability Engine:** Dynamic Grad-CAM hooked into the terminal Conv2D layer (`conv2d_3`).
            """
        )

    with col_flow:
        st.markdown("### Model Analysis Workflow")
        st.markdown(
            """
            1. **Image Ingestion:** User uploads a microscopy image or selects a curated test-set cell.
            2. **Preprocessing:** Image converted to RGB and resized to 128 × 128 without duplicate scaling.
            3. **Inference:** Forward pass yielding model probability scores $P(\\text{Parasitized})$ vs $P(\\text{Uninfected})$.
            4. **Explainability:** Dynamic backpropagation to compute Grad-CAM spatial activation maps from layer `conv2d_3`.
            5. **Saliency Analysis:** Grad-CAM overlay visualizes convolutional activation patterns associated with the model's output score.
            """
        )

    st.markdown("---")
    st.markdown("### Ready to Test?")
    st.write("Navigate to **Cell Classification** to test your own microscopy cell images, or explore **Grad-CAM Explainability**.")


# ==============================================================================
# PAGE 2: Cell Classification
# ==============================================================================
elif page == "🧪 Cell Classification":
    st.markdown("### Erythrocyte Classification")
    st.write(
        "Upload a microscopic red blood cell image (PNG, JPG, JPEG) or select one of the "
        "pre-loaded benchmark test cells to generate predictions and probability scores."
    )

    # Input mode selector
    input_source = st.radio(
        "Choose Image Source:",
        ["📤 Upload an Image", "🖼️ Select from Benchmark Sample Gallery"],
        horizontal=True,
    )

    selected_image_input = None
    sample_label_note = ""

    if input_source == "📤 Upload an Image":
        uploaded_file = st.file_uploader(
            "Upload Red Blood Cell Image",
            type=["png", "jpg", "jpeg"],
            help="Supported formats: PNG, JPG, JPEG. Images will be automatically resized to 128 × 128 RGB.",
        )
        if uploaded_file is not None:
            selected_image_input = uploaded_file
    else:
        sample_dict = get_sample_images_dict()
        if sample_dict:
            chosen_sample = st.selectbox(
                "Select a sample image from the test set:",
                list(sample_dict.keys()),
            )
            selected_image_input = sample_dict[chosen_sample]
            sample_label_note = f"Loaded sample: **{chosen_sample}**"
            st.caption(sample_label_note)
        else:
            st.warning("Sample gallery directory is empty. Please upload an image.")

    if selected_image_input is not None:
        try:
            # Preprocess image
            img_batch, pil_resized = preprocess_image(selected_image_input)

            # Store in session state for Grad-CAM tab
            st.session_state["current_img_batch"] = img_batch
            st.session_state["current_pil_img"] = pil_resized

            st.markdown("---")
            col_view, col_pred = st.columns([1, 1.3])

            with col_view:
                st.markdown("#### Input Microscopy Image")
                st.image(
                    pil_resized,
                    caption=f"Resized to Model Input ({TARGET_SIZE[0]}×{TARGET_SIZE[1]} RGB)",
                    width=220,
                )
                st.caption(
                    f"Dimensions: {pil_resized.size[0]}×{pil_resized.size[1]} | "
                    f"Channels: {len(pil_resized.getbands())} ({pil_resized.mode})"
                )

            with col_pred:
                st.markdown("#### Model Prediction")
                # Load model and run inference
                with st.spinner("Executing neural network forward pass..."):
                    model = load_trained_model()
                    pred_res = predict_cell(model, img_batch, threshold=0.5)
                    st.session_state["last_pred_res"] = pred_res

                # Display prominent badge
                render_prediction_badge(
                    predicted_class=pred_res["predicted_class"],
                    confidence_percent=pred_res["confidence_percent"],
                )

                st.caption(
                    "ℹ️ Displayed confidence is an internal model sigmoid probability, not a measure of diagnostic certainty."
                )

                # Probabilities table
                st.markdown("##### Model Class Probability Distribution")
                prob_df = pd.DataFrame(
                    {
                        "Class": ["Uninfected", "Parasitized"],
                        "Model Probability": [
                            pred_res["probability_uninfected"],
                            pred_res["probability_parasitized"],
                        ],
                        "Percentage": [
                            f"{pred_res['probability_uninfected'] * 100:.2f}%",
                            f"{pred_res['probability_parasitized'] * 100:.2f}%",
                        ],
                    }
                )
                st.table(prob_df.set_index("Class"))

                # Progress bar for parasite probability
                st.markdown(
                    "**Model Output Probability for Parasitized Class (Threshold = 0.50):**"
                )
                st.progress(pred_res["probability_parasitized"])

            st.markdown("---")
            st.markdown("#### Saliency & Explainability")
            st.write(
                "Want to inspect which cellular regions triggered this prediction? "
                "Navigate to the **🔍 Grad-CAM Explainability** section in the sidebar to visualize "
                "the convolutional feature maps for this cell."
            )

        except Exception as e:
            st.error(f"Image Processing or Classification Error: {str(e)}")
            st.info("Please verify that the uploaded file is a valid microscopy image in PNG, JPG, or JPEG format.")
    else:
        st.info("Please upload an image or select a sample from the gallery above to run classification.")


# ==============================================================================
# PAGE 3: Grad-CAM Explainability
# ==============================================================================
elif page == "🔍 Grad-CAM Explainability":
    st.markdown("### Grad-CAM Visual Explainability")
    st.write(
        "Gradient-weighted Class Activation Mapping (**Grad-CAM**) visualizes spatial activation patterns "
        "in the convolutional layers by backpropagating gradients from the model output score. "
        "Grad-CAM indicates regions that influenced the model's decision; it does not prove that the network "
        "has learned medically verified biological features of malaria parasites."
    )

    # Check if an image is already in session state
    has_active_image = (
        "current_img_batch" in st.session_state
        and "current_pil_img" in st.session_state
    )

    st.markdown("#### 1. Target Cell Selection")
    if has_active_image:
        use_active = st.checkbox(
            "Use currently loaded image from Cell Classification tab",
            value=True,
        )
    else:
        use_active = False

    if not use_active:
        sample_dict = get_sample_images_dict()
        if sample_dict:
            chosen = st.selectbox("Select a cell for Grad-CAM:", list(sample_dict.keys()))
            img_batch, pil_resized = preprocess_image(sample_dict[chosen])
            st.session_state["current_img_batch"] = img_batch
            st.session_state["current_pil_img"] = pil_resized
        else:
            uploaded = st.file_uploader(
                "Upload an image for Grad-CAM:",
                type=["png", "jpg", "jpeg"],
                key="gradcam_upload",
            )
            if uploaded:
                img_batch, pil_resized = preprocess_image(uploaded)
                st.session_state["current_img_batch"] = img_batch
                st.session_state["current_pil_img"] = pil_resized
            else:
                img_batch, pil_resized = None, None
    else:
        img_batch = st.session_state["current_img_batch"]
        pil_resized = st.session_state["current_pil_img"]

    if img_batch is not None and pil_resized is not None:
        try:
            model = load_trained_model()
            conv_layers = get_all_conv_layers(model)
            default_layer = find_last_conv_layer(model)

            # Sidebar/Controls for Grad-CAM
            st.markdown("#### 2. Visual Settings")
            c_layer, c_cmap, c_alpha = st.columns([1.2, 1, 1])

            with c_layer:
                selected_layer = st.selectbox(
                    "Target Conv2D Layer:",
                    options=conv_layers,
                    index=conv_layers.index(default_layer) if default_layer in conv_layers else len(conv_layers) - 1,
                    help="Dynamically discovered convolutional layers. The final layer provides the highest-level semantic localization.",
                )
            with c_cmap:
                selected_cmap = st.selectbox(
                    "Heatmap Colormap:",
                    ["jet", "viridis", "inferno", "plasma", "magma"],
                    index=0,
                )
            with c_alpha:
                selected_alpha = st.slider(
                    "Overlay Opacity (Alpha):",
                    min_value=0.1,
                    max_value=0.9,
                    value=0.45,
                    step=0.05,
                )

            # Compute Grad-CAM
            with st.spinner("Computing gradient backpropagation..."):
                heatmap, resolved_layer, error_msg = compute_gradcam_heatmap(
                    model=model,
                    img_batch=img_batch,
                    layer_name=selected_layer,
                )
                pred_res = predict_cell(model, img_batch)

            if error_msg:
                st.error(f"Grad-CAM Error: {error_msg}")
            else:
                colored_heatmap, blended_overlay = overlay_gradcam_on_image(
                    original_pil=pil_resized,
                    heatmap=heatmap,
                    colormap=selected_cmap,
                    alpha=selected_alpha,
                )

                st.markdown("---")
                st.markdown("#### 3. Side-by-Side Saliency Comparison")

                col_orig, col_heat, col_over = st.columns(3)

                with col_orig:
                    st.markdown("##### Original Cell (128×128)")
                    st.image(pil_resized, use_container_width=True)
                    st.caption(
                        f"Model Prediction: **{pred_res['predicted_class']}** "
                        f"(Model Probability: {pred_res['confidence_percent']:.1f}%)"
                    )

                with col_heat:
                    st.markdown(f"##### Saliency Heatmap (`{selected_cmap}`)")
                    st.image(colored_heatmap, use_container_width=True)
                    st.caption(f"Activation map from layer `{resolved_layer}` (native 16×16 upsampled to 128×128)")

                with col_over:
                    st.markdown(f"##### Superimposed Overlay (α = {selected_alpha:.2f})")
                    st.image(blended_overlay, use_container_width=True)
                    st.caption("Heatmap blended over input image to inspect activation regions")

                st.markdown("---")
                st.markdown("#### 4. Scientific & Technical Interpretation")
                explanation = explain_gradcam_findings(
                    predicted_class=pred_res["predicted_class"],
                    probability_parasitized=pred_res["probability_parasitized"],
                    layer_name=resolved_layer,
                )

                st.markdown(
                    f"""
                    <div class="info-card">
                        <strong>Model Activation Summary:</strong><br>{explanation['summary']}<br><br>
                        <strong>Scientific & Artifact Context:</strong><br>{explanation['biological_context']}<br><br>
                        <strong>Technical Resolution Context:</strong><br>{explanation['technical_context']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        except Exception as e:
            st.error(f"An unexpected error occurred during Grad-CAM visualization: {e}")

    st.markdown("---")
    st.markdown("### Benchmark Test-Set Grad-CAM Analysis")
    st.write(
        "Below is the benchmark Grad-CAM analysis across confusion matrix categories "
        "(True Positive, True Negative, False Positive, False Negative) evaluated on test-set examples."
    )
    st.info(
        "🔍 **Scientific Observation on Model Activations:**\n\n"
        "• **True Positive:** Strong activation focuses over regions exhibiting intra-cellular stain contrast.\n"
        "• **True Negative (Uninfected):** Prominent activation is distributed along the image perimeter, boundaries, and corners. "
        "This boundary activation demonstrates that model gradient sensitivity is influenced by edge transitions, background illumination, "
        "or dataset-specific cropping artifacts rather than intracellular biology.\n"
        "• **Interpretability Caution:** Grad-CAM visualizes internal model activations under its trained weights; it does not prove "
        "that the network has learned medically verified parasite features (such as ring forms, chromatin dots, or trophozoites). "
        "Furthermore, low, diffuse, or peripheral activation does not confirm the absence of parasites."
    )
    gradcam_bench_path = RESULTS_DIR / "custom_cnn_gradcam.png"
    if gradcam_bench_path.is_file():
        st.image(
            str(gradcam_bench_path),
            caption="Custom CNN Grad-CAM across Confusion Matrix Categories (Historical Test Set Evaluation)",
            use_container_width=True,
        )


# ==============================================================================
# PAGE 4: Model Performance
# ==============================================================================
elif page == "📊 Model Performance":
    st.markdown("### Model Evaluation Dashboard")
    st.write(
        "Experimental validation results on the patient-independent test set containing "
        "**5,497 microscopic cell images** from **40 completely unseen patients** within the NIH dataset."
    )
    st.info(
        "📌 **Scope of Validation:** The reported test metrics evaluate generalization to withheld patient groups "
        "within the NIH Malaria Cell Images collection. This represents internal patient-independent validation. "
        "It does not constitute external validation across different clinical environments, distinct hospital patient "
        "populations, varying staining protocols (e.g., Field or Leishman stains), or diverse microscope imaging systems. "
        "The model is an educational research prototype and is not clinically ready or approved for diagnostic use."
    )

    st.markdown("#### Primary Performance Metrics (Test Set)")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_metric_card("Accuracy", "95.23%", "5,235 / 5,497 correct")
    with col2:
        render_metric_card("Precision", "96.73%", "Low false positive rate")
    with col3:
        render_metric_card("Recall (Sens.)", "93.64%", "2,577 / 2,752 detected")
    with col4:
        render_metric_card("F1-Score", "95.16%", "Harmonic balance")
    with col5:
        render_metric_card("ROC-AUC", "98.74%", "Discrimination index")

    st.markdown("---")

    col_cm_plot, col_cm_table = st.columns([1.1, 1])

    with col_cm_plot:
        st.markdown("#### Confusion Matrix Visualization")
        cm_image_path = RESULTS_DIR / "custom_cnn_confusion_matrix.png"
        if cm_image_path.is_file():
            st.image(
                str(cm_image_path),
                caption="Custom CNN Test Set Confusion Matrix (5,497 images)",
                use_container_width=True,
            )
        else:
            st.warning("Confusion matrix plot file not found.")

    with col_cm_table:
        st.markdown("#### Confusion Matrix Breakdown")
        cm_data = pd.DataFrame(
            [
                ["Uninfected (Actual)", "2,658 (TN)", "87 (FP)"],
                ["Parasitized (Actual)", "175 (FN)", "2,577 (TP)"],
            ],
            columns=["Ground Truth", "Predicted Uninfected", "Predicted Parasitized"],
        )
        st.table(cm_data.set_index("Ground Truth"))

        st.markdown(
            """
            <div class="info-card">
                <strong>Clinical Significance of Metrics:</strong><br>
                • <strong>False Negatives (175, 6.36%):</strong> In clinical screening, false negatives are
                the most critical error: an infected patient may go untreated. The model achieved
                a high Recall of <strong>93.64%</strong>.<br>
                • <strong>False Positives (87, 3.17%):</strong> Uninfected cells flagged as infected.
                High Precision (<strong>96.73%</strong>) minimizes unnecessary confirmatory diagnostic burdens.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### Training History & Learning Curves")
    lc_image_path = RESULTS_DIR / "custom_cnn_learning_curves.png"
    if lc_image_path.is_file():
        st.image(
            str(lc_image_path),
            caption="Custom CNN Loss, Accuracy, Precision, and Recall Curves across Epochs",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("#### Comparative Benchmark: Custom CNN vs MobileNetV2")
    st.write(
        "Both models were trained and evaluated on identical patient-independent splits "
        "(16,544 train, 5,517 validation, 5,497 test images)."
    )

    comp_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "False Negatives (lower is better)"],
            "Custom CNN (Our Model)": ["95.23%", "96.73%", "93.64%", "95.16%", "98.74%", "175"],
            "MobileNetV2 (Transfer Learning)": ["92.01%", "93.30%", "90.55%", "91.90%", "97.37%", "260"],
            "Performance Delta": ["+3.22%", "+3.43%", "+3.09%", "+3.26%", "+1.37%", "-85 FN (-32.7%)"],
        }
    )
    st.table(comp_df.set_index("Metric"))

    val_comp_path = RESULTS_DIR / "model_validation_comparison.png"
    if val_comp_path.is_file():
        st.image(
            str(val_comp_path),
            caption="Validation Comparison: Custom CNN vs MobileNetV2",
            use_container_width=True,
        )


# ==============================================================================
# PAGE 5: About the Research
# ==============================================================================
elif page == "📖 About the Research":
    st.markdown("### Research Background & Methodology")

    st.markdown("#### 1. Dataset Characteristics")
    st.markdown(
        f"""
        This project utilizes the **{DATASET_NAME}** curated by the National Library of Medicine (NLM),
        part of the National Institutes of Health (NIH), Bethesda, Maryland, USA.
        - **Total segmented red blood cell images:** {DATASET_STATS['total_images']:,}
        - **Total unique patients:** {DATASET_STATS['total_patients']}
        - **Classes:** Binary (Parasitized vs Uninfected, 13,779 images per class)
        - **Resolution:** Variable original dimensions, standardized to 128 × 128 RGB.
        - **Citation:** {SOURCE_CITATION} ([NIH Public Dataset]({SOURCE_URL}))
        """
    )

    st.markdown("#### 2. Patient-Independent Splitting Strategy")
    st.write(
        """
        In medical machine learning, **data leakage** can occur when cells from the same patient smear
        are randomly partitioned across training and testing partitions. Because blood smears from the same
        patient share unique slide preparation characteristics, stain batch variations, illumination,
        and microscope optics, naive random splits allow neural networks to memorize patient-specific
        signatures rather than generalizable features.

        To reduce data leakage and evaluate generalization to unseen patients, this study partitioned
        data at the patient level using `StratifiedGroupKFold` (5 folds, random state 42, grouped by `patient_id`).
        One full fold (40 patients, 5,497 images) was held out strictly for testing, with remaining folds
        divided into training (120 patients, 16,544 images) and validation (41 patients, 5,517 images):
        """
    )

    splits_data = pd.DataFrame(
        [
            ["Training", "120", "16,544", "8,239", "8,305", "60.0%"],
            ["Validation", "41", "5,517", "2,795", "2,722", "20.0%"],
            ["Testing", "40", "5,497", "2,745", "2,752", "20.0%"],
            ["Total", "201", "27,558", "13,779", "13,779", "100.0%"],
        ],
        columns=["Partition", "Patients", "Total Cells", "Uninfected", "Parasitized", "Ratio"],
    )
    st.table(splits_data.set_index("Partition"))

    st.markdown("#### 3. Custom CNN Architecture Details")
    st.write(
        """
        The Custom CNN architecture was designed to balance representational power with
        computational compactness suitable for potential edge research deployment:
        """
    )

    arch_df = pd.DataFrame(
        [
            ["Input Layer", "Input(shape=(128, 128, 3))", "128 × 128 × 3", "Accepts standard RGB microscopic images"],
            ["Rescaling", "Rescaling(1.0 / 255)", "128 × 128 × 3", "In-graph normalization [0, 1]"],
            ["Block 1", "Conv2D(16, 3) + BatchNorm + ReLU + MaxPool(2)", "64 × 64 × 16", "Low-level spatial filtering (local edges, contrast, color transitions)"],
            ["Block 2", "Conv2D(32, 3) + BatchNorm + ReLU + MaxPool(2)", "32 × 32 × 32", "Intermediate spatial feature representation"],
            ["Block 3", "Conv2D(64, 3) + BatchNorm + ReLU + MaxPool(2)", "16 × 16 × 64", "Higher-level localized spatial pattern representation"],
            ["Block 4", "Conv2D(128, 3) + BatchNorm + ReLU", "16 × 16 × 128", "Terminal convolutional feature maps (16 × 16 grid, Grad-CAM hook)"],
            ["Global Average Pooling", "GlobalAveragePooling2D()", "128", "Spatial reduction to 1D vector; reduces parameter count"],
            ["Dropout 1", "Dropout(0.4)", "128", "Random unit deactivation to mitigate feature co-adaptation"],
            ["Dense Hidden", "Dense(64, activation='relu')", "64", "Fully connected non-linear feature projection"],
            ["Dropout 2", "Dropout(0.3)", "64", "Secondary regularization barrier"],
            ["Classification Head", "Dense(1, activation='sigmoid')", "1", "Outputs model probability P(Parasitized) ∈ [0, 1]"],
        ],
        columns=["Layer / Block", "Specification", "Output Shape", "Role & Description"],
    )
    st.table(arch_df.set_index("Layer / Block"))

    st.markdown("#### 4. Limitations & Research Disclaimer")
    st.warning(
        "Important Research Limitations & Diagnostic Boundaries:\n\n"
        "1. **Internal vs. External Validation:** The test set evaluates 5,497 images from 40 patients within the NIH dataset. "
        "While patient-level grouping reduces data leakage within this collection, this does not constitute external clinical validation. "
        "Generalization to external hospitals, diverse patient demographics, alternate staining chemistries (e.g., Field or Leishman stain), "
        "and different microscope optics has not been demonstrated.\n\n"
        "2. **No Clinical Readiness:** This software is strictly an educational research prototype. It is not an FDA-cleared, "
        "CE-marked, or clinically validated diagnostic device, and must not be used for patient diagnosis or treatment planning.\n\n"
        "3. **Confounding Artifacts:** Automated image-level classifiers may respond to slide debris, staining precipitates, "
        "Howell-Jolly bodies, platelets overlying erythrocytes, or abnormal RBC morphology rather than true parasites.\n\n"
        "4. **Clinical Standard:** Definitive malaria diagnosis requires comprehensive clinical evaluation of multiple microscopic fields "
        "across thick and thin blood smears by a qualified medical laboratory professional."
    )
