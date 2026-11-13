"""
UI styles, CSS theming, and reusable visual components for the Streamlit app.
"""

import streamlit as st
from app.metrics_data import DISCLAIMER_TEXT


def apply_custom_css():
    """
    Inject custom CSS for a polished, modern medical-AI research aesthetic.
    Supports clean spacing, responsive metric cards, and crisp typography.
    """
    st.markdown(
        """
        <style>
        /* Import clean font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        /* Top header container */
        .research-header {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
            padding: 1.8rem 2rem;
            border-radius: 12px;
            color: #ffffff;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 14px 0 rgba(0, 0, 0, 0.12);
        }

        .research-header h1 {
            color: #ffffff !important;
            font-size: 1.75rem;
            font-weight: 700;
            margin: 0 0 0.4rem 0;
            letter-spacing: -0.02em;
        }

        .research-header p {
            color: #94a3b8;
            font-size: 0.95rem;
            margin: 0;
            line-height: 1.5;
        }

        .tag-pill {
            display: inline-block;
            background: rgba(20, 184, 166, 0.2);
            color: #2dd4bf;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            margin-bottom: 0.6rem;
            border: 1px solid rgba(45, 212, 191, 0.3);
        }

        /* Metric cards */
        .metric-card-container {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 1.1rem 1.25rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            height: 100%;
        }

        @media (prefers-color-scheme: dark) {
            .metric-card-container {
                background: #1e293b;
                border-color: #334155;
            }
        }

        .metric-card-label {
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            margin-bottom: 0.25rem;
        }

        .metric-card-value {
            font-size: 1.85rem;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.2;
        }

        @media (prefers-color-scheme: dark) {
            .metric-card-value {
                color: #f8fafc;
            }
        }

        .metric-card-subtext {
            font-size: 0.75rem;
            color: #0d9488;
            font-weight: 500;
            margin-top: 0.2rem;
        }

        /* Medical Disclaimer box */
        .disclaimer-box {
            background: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 0.9rem 1.1rem;
            border-radius: 6px;
            margin: 1rem 0 1.5rem 0;
            color: #78350f;
            font-size: 0.825rem;
            line-height: 1.45;
        }

        @media (prefers-color-scheme: dark) {
            .disclaimer-box {
                background: #292524;
                border-left-color: #f59e0b;
                color: #fef3c7;
            }
        }

        /* Prediction banner */
        .prediction-banner {
            padding: 1.3rem 1.6rem;
            border-radius: 10px;
            margin: 1rem 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .banner-parasitized {
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            border: 1.5px solid #f87171;
            color: #991b1b;
        }

        @media (prefers-color-scheme: dark) {
            .banner-parasitized {
                background: linear-gradient(135deg, #450a0a 0%, #2b0505 100%);
                border-color: #ef4444;
                color: #fecaca;
            }
        }

        .banner-uninfected {
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border: 1.5px solid #4ade80;
            color: #166534;
        }

        @media (prefers-color-scheme: dark) {
            .banner-uninfected {
                background: linear-gradient(135deg, #052e16 0%, #021a0c 100%);
                border-color: #22c55e;
                color: #bbf7d0;
            }
        }

        .pred-title {
            font-size: 1.4rem;
            font-weight: 700;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .pred-conf {
            font-size: 1.1rem;
            font-weight: 600;
            margin-top: 0.2rem;
        }

        /* Info card */
        .info-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin-bottom: 1rem;
        }

        @media (prefers-color-scheme: dark) {
            .info-card {
                background: #1e293b;
                border-color: #334155;
            }
        }

        /* Section header badge */
        .section-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 1.25rem;
            font-weight: 700;
            color: #0f172a;
            margin: 1.2rem 0 0.8rem 0;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 0.4rem;
        }

        @media (prefers-color-scheme: dark) {
            .section-header {
                color: #f1f5f9;
                border-bottom-color: #334155;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str, tag: str = "Research Prototype"):
    """Render top stylized application banner."""
    st.markdown(
        f"""
        <div class="research-header">
            <div class="tag-pill">{tag}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer():
    """Render persistent medical research disclaimer."""
    st.markdown(
        f"""
        <div class="disclaimer-box">
            <strong>⚠️ Research & Educational Notice:</strong> {DISCLAIMER_TEXT}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, subtext: str = ""):
    """Render attractive individual metric card."""
    st.markdown(
        f"""
        <div class="metric-card-container">
            <div class="metric-card-label">{label}</div>
            <div class="metric-card-value">{value}</div>
            {f'<div class="metric-card-subtext">{subtext}</div>' if subtext else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_badge(predicted_class: str, confidence_percent: float):
    """Render high-clarity status banner for classification outcome."""
    is_parasitized = (predicted_class == "Parasitized")
    banner_class = "banner-parasitized" if is_parasitized else "banner-uninfected"
    icon = "🩸" if is_parasitized else "🛡️"
    status_label = "PARASITIZED (Infected)" if is_parasitized else "UNINFECTED (Healthy)"
    subtext = (
        "The model classified this image as Parasitized"
        if is_parasitized
        else "The model classified this image as Uninfected"
    )

    st.markdown(
        f"""
        <div class="prediction-banner {banner_class}">
            <div>
                <div class="pred-title">{icon} {status_label}</div>
                <div style="font-size: 0.85rem; opacity: 0.9; margin-top: 0.2rem;">{subtext}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.8;">Model Probability</div>
                <div class="pred-conf">{confidence_percent:.2f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
