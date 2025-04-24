"""
Streamlit-ready Stress-Ribbon / Catenary Cable Profiler  –  R0.5

• Replicates the Jupyter/ipython widget version but uses Streamlit widgets.
• Adds a user-set Design Factor ϕ (default 0.6) for realistic horizontal tension.
• Includes σ_tu utilisation % and keeps optional ML sag prediction.

Author credits appear in the corner of each plot and in the page footer.
"""

# ============= 1. Imports ====================================================
import os
import math
import csv
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – used implicitly by mpl
from sklearn.ensemble import RandomForestRegressor

import streamlit as st

# Silence sklearn warnings during demos
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

CREDIT_TEXT = "© Vijaykumar Parmar & Dr. K. B. Parikh"
DATA_DIR = Path(__file__).parent

# ============= 2. Helpers ====================================================

def get_builtin_data():
    """Return ~100 synthetic rows for boot-strapping the RF model."""
    spans = [40, 60, 80, 100, 120]
    extended_spans = list(range(100, 1300, 100))
    udls = [10, 15, 20]
    cables = [2, 3, 4]
    dias = [12, 14, 16]
    strength = 1860  # MPa
    spacing_fixed = 1.5  # m

    base = list(itertools.product(spans, udls, cables, dias))[:30]
    ext = list(itertools.product(extended_spans, udls, cables, dias))[:70]

    rows = []
    for L, w, n, d in base + ext:
        area = math.pi * (d / 2) ** 2
        H = n * area * strength / 1000  # kN
        sag = (w * L ** 2) / (8 * H)
        rows.append([L, w, n, spacing_fixed, d, strength, round(sag, 3), 90])

    cols = ["Span", "UDL", "No. Cables", "Spacing", "Dia", "Strength", "Sag", "Feedback"]
    return pd.DataFrame(rows, columns=cols)


@st.cache_resource(show_spinner=False)
def train_ml_model(feedback_path: Path | None = None):
    """Train a RandomForest on synthetic + user-feedback rows."""
    df_builtin = get_builtin_data()

    # --- user feedback --------------------------------------------------
    if feedback_path and feedback_path.exists() and feedback_path.stat().st_size:
        df_fb_raw = pd.read_csv(feedback_path)
        good_cols = ["Span", "UDL", "No. Cables", "Spacing", "Dia", "Strength", "Sag", "Feedback"]
        if list(df_fb_raw.columns) == good_cols:
            df_fb = df_fb_raw[df_fb_raw["Feedback"] > 60]
        else:
            df_fb = pd.DataFrame()
    else:
        df_fb = pd.DataFrame()

    df_train = pd.concat([df_builtin, df_fb], ignore_index=True)
    if len(df_train) < 10:
        return None, 0

    X = df_train[["Span", "UDL", "No. Cables", "Spacing", "Dia", "Strength"]].fillna(df_train.median(numeric_only=True))
    y = df_train["Sag"].fillna(df_train["Sag"].median())

    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, min_samples_split=5)
    model.fit(X, y)
    return model, len(df_train) - len(df_builtin)


# ============= 3. Streamlit UI ==============================================
st.set_page_config(page_title="Stress-Ribbon Cable Profiler", layout="wide")
st.title("Stress-Ribbon / Catenary Cable Profiler")

with st.sidebar:
    st.header("Input Parameters")
    L = st.number_input("Span L (m)", value=50.0, min_value=1.0, step=1.0)
    w = st.number_input("UDL w (kN/m)", value=5.0, min_value=0.0, step=0.5)
    n = st.number_input("Number of Cables n", value=2, min_value=1, step=1, format="%d")
    spacing = st.number_input("Cable Spacing (m)", value=1.0, min_value=0.0, step=0.1)
    dia_mm = st.number_input("Cable Diameter d (mm)", value=20.0, min_value=1.0, step=0.5)
    strength = st.number_input("Tensile Strength σ_tu (MPa)", value=1600.0, min_value=1.0, step=50.0)
    design_factor = st.number_input("Design Factor ϕ", value=0.6, min_value=0.01, max_value=1.0, step=0.05)
    mode = st.radio("Mode", ("Mathematical Only", "Mathematical + AI"))
    calc = st.button("Generate Profile & Calculate", type="primary")

# place-holders for results and plots
res_container = st.container()
plot_container = st.container()
plot3d_container = st.container()

feedback_path = DATA_DIR / "feedback_log.csv"

# ============= 4. Calculation ===============================================
if calc:
    # --- basic validation --------------------------------------------------
    if n == 1:
        spacing = 0.0
    area_mm2 = math.pi * (dia_mm / 2) ** 2
    H_kN = n * area_mm2 * (design_factor * strength) / 1000  # kN
    if H_kN == 0:
        st.error("Horizontal tension computed as 0 kN – check inputs.")
        st.stop()

    sag_eng = (w * L ** 2) / (8 * H_kN)
    sag_ratio = sag_eng / L * 100
    V_kN = w * L / 2
    T_kN = math.hypot(H_kN, V_kN)
    sigma_actual = (T_kN * 1000) / (n * area_mm2)
    utilisation = sigma_actual / strength * 100

    # Optional AI sag prediction
    sag_ml = None
    sag_ratio_ml = None
    feedback_rows = 0
    if mode == "Mathematical + AI":
        with st.spinner("Training / loading ML model …"):
            model, feedback_rows = train_ml_model(feedback_path)
        if model is not None:
            Xnew = pd.DataFrame([[L, w, n, spacing, dia_mm, strength]],
                                columns=["Span", "UDL", "No. Cables", "Spacing", "Dia", "Strength"])
            sag_ml = float(model.predict(Xnew)[0])
            if 0 < sag_ml < L:
                sag_ratio_ml = sag_ml / L * 100
            else:
                sag_ml = None

    # --- Results table -----------------------------------------------------
    with res_container:
        st.subheader("Results")
        st.markdown(
            f"""
            | Parameter | Value |
            |-----------|----------------:|
            | Cable Area (single) | {area_mm2:,.2f} mm² |
            | Horizontal Tension H | {H_kN:,.2f} kN |
            | Vertical Reaction V | {V_kN:,.2f} kN |
            | Sag f (Math) | {sag_eng:,.3f} m ({sag_ratio:.2f} % of span) |
            | Support Tension T | {T_kN:,.2f} kN |
            | Actual Stress σ | {sigma_actual:,.2f} MPa |
            | Utilisation | {utilisation:5.1f} % of σ_tu |
            """,
            unsafe_allow_html=True,
        )
        if sag_ml is not None:
            st.info(f"AI Sag f_ml: {sag_ml:.3f} m ({sag_ratio_ml:.2f} % of span) ― trained on {feedback_rows} feedback rows.")
        if sigma_actual > strength:
            st.error("σ_actual exceeds σ_tu! (unsafe)")
        else:
            st.success("σ_actual within limit.")

    # --- 2-D plot -----------------------------------------------------------
    x = np.linspace(0, L, 200)
    y_eng = (4 * sag_eng / L ** 2) * x * (L - x)
    y_ml = (4 * sag_ml / L ** 2) * x * (L - x) if sag_ml else None

    fig2d, ax2d = plt.subplots(figsize=(8, 4))
    ax2d.plot(x, y_eng, "--", label=f"Eng (f={sag_eng:.3f} m)")
    if y_ml is not None:
        ax2d.plot(x, y_ml, label=f"AI  (f={sag_ml:.3f} m)")
    ax2d.set(title=f"Stress-Ribbon Profile — L={L} m", xlabel="Length (m)", ylabel="Sag (m)")
    ylim = max(sag_eng, sag_ml or 0) * 1.1
    ax2d.set_ylim(ylim, -0.1 * ylim)
    ax2d.grid(ls=":", alpha=.7)
    ax2d.legend()
    ax2d.text(0.99, 0.01, CREDIT_TEXT, ha="right", va="bottom", fontsize=6, style="italic", alpha=.7, transform=ax2d.transAxes)
    plot_container.pyplot(fig2d, use_container_width=True)

    # --- 3-D plot (optional) -----------------------------------------------
    fig3d = plt.figure(figsize=(9, 6))
    ax3d = fig3d.add_subplot(111, projection="3d")

    y_positions = np.linspace(-(n - 1) * spacing / 2, (n - 1) * spacing / 2, n) if n > 1 else [0]
    z_curve = y_ml if y_ml is not None else y_eng
    for idx, y0 in enumerate(y_positions, start=1):
        ax3d.plot(x, np.full_like(x, y0), z_curve, label=f"Cable {idx}")
    ax3d.set(title=f"3-D Layout (n={n}, s={spacing} m)", xlabel="Length (m)", ylabel="Transverse (m)", zlabel="Sag (m)")
    ax3d.set_zlim(max(z_curve) * 1.1, 0)
    ax3d.view_init(20, -50)
    if n > 1:
        ax3d.legend(loc="upper left", bbox_to_anchor=(0.85, 1.0))
    fig3d.text(0.99, 0.01, CREDIT_TEXT, ha="right", va="bottom", fontsize=6, style="italic", alpha=.7)
    plot3d_container.pyplot(fig3d, use_container_width=True)

    # --- Feedback (optional) ----------------------------------------------
    with st.expander("💬 Give feedback for ML (optional)"):
        rating = st.slider("Rate Output (%)", 0, 100, 80)
        if st.button("Submit Feedback"):
            feedback_row = [L, w, n, spacing, dia_mm, strength, round(sag_eng, 3), rating]
            header_needed = not feedback_path.exists() or feedback_path.stat().st_size == 0
            with open(feedback_path, "a", newline="") as f:
                writer = csv.writer(f)
                if header_needed:
                    writer.writerow(["Span", "UDL", "No. Cables", "Spacing", "Dia", "Strength", "Sag", "Feedback"])
                writer.writerow(feedback_row)
            st.success("Feedback saved – thank you!")

# ============= Footer ========================================================
st.markdown(f"<div style='text-align:right;font-size:10px;font-style:italic;'>{CREDIT_TEXT}</div>", unsafe_allow_html=True)
