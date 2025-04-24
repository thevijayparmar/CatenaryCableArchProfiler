# =============
#  Stress-Ribbon / Catenary Cable Profiler  –  Streamlit edition
#
#  ⚠️  NOTE TO FUTURE MAINTAINERS: Written at multiple mid-night sessions with coffee in hand.
#      If something looks weird, blame the caffeine, not the code.
#
#  ✍️  Authors: Vijaykumar Parmar & Dr. K. B. Parikh  (2025)
# ==============

# 1. Imports (only Streamlit-friendly libs)
import streamlit as st
import numpy as np
import pandas as pd
import csv, os, warnings, itertools, math
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D        # noqa: F401 – implicit use by mpl

warnings.filterwarnings("ignore", category=UserWarning,  module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

CREDIT_TEXT = "© Vijaykumar Parmar & Dr. K. B. Parikh"

# ----------------------------------------------------------------------
# 2.   Streamlit UI – replaces the ipywidgets section
# ----------------------------------------------------------------------
st.set_page_config(page_title="Stress-Ribbon / Catenary Cable Profiler",
                   layout="wide")

st.title("Stress-Ribbon / Catenary Cable Profiler")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    L        = st.number_input("Span L (m)",            min_value=0.0,  value=50.0, step=1.0)
    w        = st.number_input("UDL w (kN/m)",          min_value=0.0,  value=5.0,  step=0.5)
    n        = st.number_input("No. of Cables n",       min_value=1,    value=2,    step=1)

with col2:
    spacing  = st.number_input("Cable Spacing (m)",     min_value=0.0,  value=1.0,  step=0.1)
    dia_mm   = st.number_input("Cable Dia d (mm)",      min_value=0.1,  value=20.0, step=1.0)
    strength = st.number_input("Tensile σ_tu (MPa)",    min_value=1.0,  value=1600.0, step=10.0)

with col3:
    design_factor = st.number_input("Design Factor ϕ",  min_value=0.0,  max_value=1.0, value=0.6, step=0.05)
    mode          = st.radio("Mode", ["Mathematical Only", "Mathematical + AI"])
    generate_btn  = st.button("Generate Profile & Calculate", type="primary")
    feedback_val  = st.slider("Rate Output (%)", 0, 100, 80)
    save_fb_btn   = st.button("Submit Feedback", disabled=True)

# ----------------------------------------------------------------------
# 3.  Globals (kept as in original)
# ----------------------------------------------------------------------
global_sag          = 0.0
ml_model            = None
feedback_data_count = 0
last_inputs         = {}

# ----------------------------------------------------------------------
# 4.  Synthetic-data generator  (unchanged)
# ----------------------------------------------------------------------
def get_builtin_data():
    spans           = [40, 60, 80, 100, 120]
    extended_spans  = list(range(100, 1300, 100))          # up to 1200 m
    udls            = [10, 15, 20]
    cables          = [2, 3, 4]
    dias            = [12, 14, 16]
    strength        = 1860                                  # MPa (fixed)
    spacing_fixed   = 1.5                                   # m

    base_combos = list(itertools.product(spans, udls, cables, dias))[:30]
    ext_combos  = list(itertools.product(extended_spans, udls, cables, dias))[:70]
    all_combos  = base_combos + ext_combos

    rows = []
    for L_, w_, n_, d_ in all_combos:
        area = np.pi * (d_ / 2) ** 2
        H    = (n_ * area * strength) / 1000                # kN
        sag  = (w_ * L_ ** 2) / (8 * H)
        rows.append([L_, w_, n_, spacing_fixed, d_, strength,
                     round(sag, 3), 90])                    # optimistic 90 %

    cols = ['Span', 'UDL', 'No. Cables', 'Spacing',
            'Dia', 'Strength', 'Sag', 'Feedback']
    return pd.DataFrame(rows, columns=cols)

# ----------------------------------------------------------------------
# 5.  Model training  (unchanged – prints go to Streamlit log)
# ----------------------------------------------------------------------
def train_ml_model():
    global feedback_data_count
    try:
        df_builtin = get_builtin_data()
        feedback_data_count = 0

        if os.path.exists("feedback_log.csv"):
            df_fb_raw = pd.read_csv("feedback_log.csv")
            expected_cols = ['Span','UDL','No. Cables','Spacing',
                             'Dia','Strength','Sag','Feedback']
            if list(df_fb_raw.columns) == expected_cols:
                df_fb = df_fb_raw[df_fb_raw['Feedback'] > 60]
                feedback_data_count = len(df_fb)
            else:
                st.warning("Feedback CSV column mismatch – ignored.")
        else:
            df_fb = pd.DataFrame()

        df_train = pd.concat([df_builtin, df_fb], ignore_index=True)
        if len(df_train) < 10:
            st.warning("Too little data (<10) for ML training.")
            return None

        X = df_train[['Span','UDL','No. Cables','Spacing','Dia','Strength']]
        y = df_train['Sag']
        X, y = X.fillna(X.median()), y.fillna(y.median())

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        ).fit(X, y)

        return model

    except Exception as err:
        st.error(f"ML training failed: {err}")
        return None

# ----------------------------------------------------------------------
# 6.  Main calculation block – runs when user clicks *Generate*
# ----------------------------------------------------------------------
if generate_btn:
    # ---------- sanity-range checks ------------------------------
    LIMITS = {
        "Span L":        (1, 2_000),
        "UDL w":         (0, 200),
        "n":             (1, 50),
        "spacing":       (0, 20),
        "diameter":      (1, 200),
        "σ_tu":          (100, 5_000),
    }
    v_bad = []
    if not (LIMITS["Span L"][0]      <= L        <= LIMITS["Span L"][1]):      v_bad.append("Span L")
    if not (LIMITS["UDL w"][0]       <= w        <= LIMITS["UDL w"][1]):       v_bad.append("UDL w")
    if not (LIMITS["n"][0]           <= n        <= LIMITS["n"][1]):           v_bad.append("n")
    if not (LIMITS["spacing"][0]     <= spacing  <= LIMITS["spacing"][1]):     v_bad.append("spacing")
    if not (LIMITS["diameter"][0]    <= dia_mm   <= LIMITS["diameter"][1]):    v_bad.append("diameter")
    if not (LIMITS["σ_tu"][0]        <= strength <= LIMITS["σ_tu"][1]):        v_bad.append("σ_tu")
    if v_bad:
        st.error("🚧 Input(s) out of realistic range: " + ", ".join(v_bad))
        st.stop()
    if not (0 < design_factor <= 1):
        st.error("Design Factor must be within (0, 1].")
        st.stop()

    if n == 1:
        spacing = 0.0

    # ---------- engineering maths -------------------------------
    area_mm2 = math.pi * (dia_mm / 2) ** 2
    H_kN     = n * area_mm2 * (design_factor * strength) / 1000
    sag_eng  = (w * L ** 2) / (8 * H_kN)
    V_kN     = w * L / 2
    T_kN     = math.hypot(H_kN, V_kN)
    σ_actual = T_kN * 1000 / (n * area_mm2)
    util     = σ_actual / strength * 100

    # overflow / NaN guard
    for val, lab in [(sag_eng,"sag"), (H_kN,"H"), (σ_actual,"σ_actual")]:
        if (not math.isfinite(val)) or abs(val) > 1e9:
            st.error(f"😵 Calculation blew up – {lab} became {val}. Check inputs.")
            st.stop()

    # ---------- optional AI sag -------------------------------
    sag_ml = None
    if mode == "Mathematical + AI":
        with st.spinner("Training / loading ML model…"):
            ml_model = train_ml_model()
        if ml_model is not None:
            try:
                sag_ml = ml_model.predict(
                    pd.DataFrame([[L, w, n, spacing, dia_mm, strength]],
                                 columns=['Span','UDL','No. Cables',
                                          'Spacing','Dia','Strength'])
                )[0]
                if sag_ml <= 0 or sag_ml > L:
                    st.warning(f"AI sag {sag_ml:.2f} m looks fishy – ignored.")
                    sag_ml = None
            except Exception as e:
                st.warning(f"AI prediction failed: {e}")

    # ---------- results table ----------------------------------
    res_tbl = pd.DataFrame({
        "Parameter": ["Cable area (single)", "Horizontal Tension H",
                      "Vertical Reaction V", "Sag f (Math)",
                      "Support Tension T",  "Actual Stress σ",
                      "Utilisation %"],
        "Value":     [f"{area_mm2:,.2f} mm²",
                      f"{H_kN:,.2f} kN",
                      f"{V_kN:,.2f} kN",
                      f"{sag_eng:,.3f} m   ({sag_eng/L*100:.2f} % of span)",
                      f"{T_kN:,.2f} kN",
                      f"{σ_actual:,.2f} MPa",
                      f"{util:5.1f} %"]
    })
    st.table(res_tbl)

    if σ_actual > strength:
        st.error("⚠️ σ_actual exceeds σ_tu! (unsafe)")
    else:
        st.success("✅ σ_actual within σ_tu")

    if sag_ml is not None:
        st.info(f"🤖 AI sag f_ml = {sag_ml:.3f} m "
                f"({sag_ml/L*100:.2f} % of span) – model trained on "
                f"{feedback_data_count} feedback rows")

    # ---------- 2-D profile plot --------------------------------
    x = np.linspace(0, L, 100)
    y_eng = (4 * sag_eng / L ** 2) * x * (L - x)
    y_ml  = (4 * sag_ml  / L ** 2) * x * (L - x) if sag_ml else None

    fig2d, ax2d = plt.subplots(figsize=(10,4))
    ax2d.plot(x, y_eng, '--', label=f'Eng (f={sag_eng:.3f} m)')
    if y_ml is not None:
        ax2d.plot(x, y_ml, label=f'AI  (f={sag_ml:.3f} m)')
    ax2d.set(title=f"Stress-Ribbon Profile – L={L} m",
             xlabel="Length (m)", ylabel="Sag (m)")
    ylim = max(sag_eng, sag_ml or 0) * 1.1
    ax2d.set_ylim(ylim, -0.1*ylim)
    ax2d.grid(':', alpha=.7)
    ax2d.legend()
    fig2d.text(0.99, 0.01, CREDIT_TEXT,
               ha='right', va='bottom', fontsize=6, style='italic', alpha=.7)
    st.pyplot(fig2d)

    # ---------- 3-D plot ---------------------------------------
    plot_sag = sag_ml if sag_ml else sag_eng
    if plot_sag > 0:
        fig3d = plt.figure(figsize=(10,6))
        ax3d  = fig3d.add_subplot(111, projection='3d')

        y_pos = np.linspace(-(n-1)*spacing/2, (n-1)*spacing/2, n) if n > 1 else [0]
        z_crv = y_ml if y_ml is not None else y_eng
        for i, y0 in enumerate(y_pos, start=1):
            ax3d.plot(x, np.full_like(x, y0), z_crv, label=f"Cable {i}")
        ax3d.set_title(f"3-D Layout (n = {n}, s = {spacing} m)")
        ax3d.set_xlabel("Length (m)")
        ax3d.set_ylabel("Transverse (m)")
        ax3d.set_zlabel("Sag (m)")
        ax3d.set_zlim(plot_sag*1.1, 0)
        ax3d.view_init(20, -50)
        if n > 1:
            ax3d.legend(loc='upper left', bbox_to_anchor=(0.85, 1.0))
        fig3d.text(0.99, 0.01, CREDIT_TEXT,
                   ha='right', va='bottom', fontsize=6, style='italic', alpha=.7)
        st.pyplot(fig3d)

    # ---------- enable feedback button and store state ----------
    last_inputs = dict(Span=L, UDL=w, **{"No. Cables": n,
                                         "Spacing": spacing,
                                         "Dia": dia_mm,
                                         "Strength": strength})
    global_sag = sag_eng
    st.session_state["allow_feedback"] = True

# ----------------------------------------------------------------------
# 7.  Feedback saver – runs when *Submit Feedback* is clicked
# ----------------------------------------------------------------------
if save_fb_btn:
    if not st.session_state.get("allow_feedback"):
        st.warning("Run a calculation before saving feedback.")
        st.stop()

    row = [last_inputs.get(k, 0) for k in
           ["Span", "UDL", "No. Cables", "Spacing", "Dia", "Strength"]]
    row += [round(global_sag, 3), feedback_val]

    file_path   = "feedback_log.csv"
    need_hdr    = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
    try:
        with open(file_path, "a", newline="") as f:
            writer = csv.writer(f)
            if need_hdr:
                writer.writerow(["Span","UDL","No. Cables","Spacing",
                                 "Dia","Strength","Sag","Feedback"])
            writer.writerow(row)
        st.success(f"Feedback {feedback_val}% saved (Sag = {global_sag:.3f} m).")
    except Exception as err:
        st.error(f"Feedback save failed: {err}")

# ----------------------------------------------------------------------
# 8.  Tiny credit in footer
# ----------------------------------------------------------------------
st.markdown(f"<div style='text-align:right; font-size:8px; "
            f"font-style:italic;'>{CREDIT_TEXT}</div>", unsafe_allow_html=True)
