# =============
#  Stress-Ribbon / Catenary Cable Profiler  –  Streamlit edition
#
#  Credit: Vijaykumar Parmar & Dr. K. B. Parikh (2025)
# =============

import streamlit as st
import numpy as np
import pandas as pd
import csv, os, warnings, itertools, math
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from mpl_toolkits.mplot3d import Axes3D        # noqa: F401 – used implicitly

warnings.filterwarnings("ignore", category=UserWarning,  module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

CREDIT_TEXT = "© Vijaykumar Parmar & Dr. K. B. Parikh"

# ------------- 4. Synthetic data for AI (unchanged) -------------------------
def get_builtin_data():
    spans           = [40, 60, 80, 100, 120]
    extended_spans  = list(range(100, 1300, 100))          # up to 1200 m
    udls            = [10, 15, 20]
    cables          = [2, 3, 4]
    dias            = [12, 14, 16]
    strength        = 1860                                  # MPa
    spacing_fixed   = 1.5                                   # m

    base_combos = list(itertools.product(spans, udls, cables, dias))[:30]
    ext_combos  = list(itertools.product(extended_spans, udls, cables, dias))[:70]
    all_combos  = base_combos + ext_combos

    rows = []
    for L, w, n, d in all_combos:
        area = np.pi * (d / 2) ** 2
        H    = n * area * strength / 1000                   # kN
        sag  = (w * L**2) / (8 * H)
        rows.append([L, w, n, spacing_fixed, d, strength, round(sag, 3), 90])

    cols = ["Span", "UDL", "No. Cables", "Spacing",
            "Dia", "Strength", "Sag", "Feedback"]
    return pd.DataFrame(rows, columns=cols)

# ------------- 5. Train / reload RF model (unchanged) -----------------------
def train_ml_model():
    global feedback_data_count
    feedback_data_count = 0

    df_builtin = get_builtin_data()
    if os.path.exists("feedback_log.csv"):
        df_fb_raw = pd.read_csv("feedback_log.csv")
        expected_cols = ['Span','UDL','No. Cables','Spacing',
                         'Dia','Strength','Sag','Feedback']
        if list(df_fb_raw.columns) == expected_cols:
            df_fb = df_fb_raw[df_fb_raw["Feedback"] > 60]
            feedback_data_count = len(df_fb)
        else:
            df_fb = pd.DataFrame()
    else:
        df_fb = pd.DataFrame()

    df_train = pd.concat([df_builtin, df_fb], ignore_index=True)
    if len(df_train) < 10:
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

# --------------------------- UI  -------------------------------------------
st.set_page_config(page_title="Stress-Ribbon / Catenary Cable Profiler",
                   layout="wide")

st.title("Stress-Ribbon / Catenary Cable Profiler")

col1, col2, col3 = st.columns(3)

with col1:
    L        = st.number_input("Span L (m)",        1.0, 2000.0, 50.0)
    w        = st.number_input("UDL w (kN/m)",      0.0, 200.0, 5.0, step=0.1)
    n        = st.number_input("No. of Cables n",   1,   50,    2)

with col2:
    spacing  = st.number_input("Cable Spacing (m)", 0.0, 20.0, 1.0, step=0.1)
    dia_mm   = st.number_input("Cable Dia d (mm)",  1.0, 200.0, 20.0, step=0.1)
    strength = st.number_input("Tensile σ_tu (MPa)",100.0, 5000.0, 1600.0, step=10.0)

with col3:
    design_factor = st.number_input("Design Factor ϕ", 0.01, 1.0, 0.6, step=0.01)
    mode          = st.radio("Mode", ["Mathematical Only", "Mathematical + AI"])

calc_clicked   = st.button("Generate Profile & Calculate", type="primary")
rating         = st.slider("Rate Output (%)", 0, 100, 80)
feedback_click = st.button("Submit Feedback")

# --------------------------- 6. Core Calculation ---------------------------
def run_calculation():
    # --- sanity limits -----------------------------------------------------
    limits = {
        "Span L":        (1, 2000),
        "UDL w":         (0, 200),
        "n":             (1, 50),
        "spacing":       (0, 20),
        "dia":           (1, 200),
        "σ_tu":          (100, 5000),
    }
    bad = []
    if not (limits["Span L"][0] <= L <= limits["Span L"][1]):           bad.append("Span L")
    if not (limits["UDL w"][0]  <= w <= limits["UDL w"][1]):            bad.append("UDL w")
    if not (limits["n"][0]      <= n <= limits["n"][1]):                bad.append("n")
    if not (limits["spacing"][0]<= spacing <= limits["spacing"][1]):    bad.append("spacing")
    if not (limits["dia"][0]    <= dia_mm <= limits["dia"][1]):         bad.append("diameter")
    if not (limits["σ_tu"][0]   <= strength <= limits["σ_tu"][1]):      bad.append("σ_tu")

    if bad:
        st.error("🚧 **Input out of realistic range:** " + ", ".join(bad))
        st.stop()

    if not (0 < design_factor <= 1):
        st.error("Design Factor must be 0 < ϕ ≤ 1")
        st.stop()

    if n == 1 and spacing != 0:
        st.info("n = 1 ⇒ spacing set to 0")
        spacing = 0

    # --- engineering math --------------------------------------------------
    area_mm2   = math.pi * (dia_mm / 2)**2
    H_kN       = n * area_mm2 * (design_factor * strength) / 1000
    sag_eng    = (w * L**2) / (8 * H_kN)
    sag_ratio  = sag_eng / L * 100
    V_kN       = w * L / 2
    T_kN       = math.hypot(H_kN, V_kN)
    sigma_act  = T_kN * 1000 / (n * area_mm2)
    utilization = sigma_act / strength * 100

    # overflow / NaN guard
    for v, tag in [(sag_eng, "sag"), (H_kN, "horizontal tension"), (sigma_act, "σ_actual")]:
        if (not math.isfinite(v)) or abs(v) > 1e9:
            st.error(f"😵 Calculation blew up – {tag} = {v}. Check inputs.")
            st.stop()

    # --- optional AI -------------------------------------------------------
    sag_ml = sag_ratio_ml = None
    if mode == "Mathematical + AI":
        with st.spinner("Training / loading ML model …"):
            model = train_ml_model()
        if model is not None:
            Xnew = pd.DataFrame([[L, w, n, spacing, dia_mm, strength]],
                                columns=['Span','UDL','No. Cables',
                                         'Spacing','Dia','Strength'])
            try:
                sag_ml = model.predict(Xnew)[0]
                if sag_ml <= 0 or sag_ml > L:
                    st.warning(f"AI sag {sag_ml:.2f} m looks unreasonable → ignored")
                    sag_ml = None
                else:
                    sag_ratio_ml = sag_ml / L * 100
            except Exception as e:
                st.warning(f"AI prediction failed: {e}")

    # ------------- RESULTS  -------------------------------------------------
    lines = [
        "--- RESULTS ---",
        f"L = {L} m | w = {w} kN/m | n = {n} | d = {dia_mm} mm "
        f"| σ_tu = {strength} MPa | s = {spacing} m | ϕ = {design_factor}",
        f"Cable area (single)     : {area_mm2:8.2f} mm²",
        f"Horizontal tension H    : {H_kN:8.2f} kN",
        f"Vertical reaction V     : {V_kN:8.2f} kN",
        "--------------------------------",
        f"🧮 Math sag f_eng        : {sag_eng:8.3f} m ({sag_ratio:.2f}% of span)",
        f"Approx. support tension : {T_kN:8.2f} kN",
        f"Actual stress σ         : {sigma_act:8.2f} MPa",
        f"Utilisation             : {utilization:6.1f}% of σ_tu",
        "⚠️  σ exceeds σ_tu!" if sigma_act > strength else "✅  σ within limit.",
    ]
    if sag_ml is not None:
        lines += [
            "--------------------------------",
            f"🤖 AI sag f_ml           : {sag_ml:8.3f} m ({sag_ratio_ml:.2f}% of span)",
            f"(trained on {feedback_data_count} feedback rows)",
        ]
    st.code("\n".join(lines), language="")

    # ------------- 2-D plot -------------------------------------------------
    x = np.linspace(0, L, 200)
    y_eng = (4 * sag_eng / L**2) * x * (L - x)
    y_ml  = (4 * sag_ml / L**2) * x * (L - x) if sag_ml else None

    plt.figure(figsize=(10, 4))
    plt.plot(x, y_eng, "--", label=f"Eng (f={sag_eng:.3f} m)")
    if y_ml is not None:
        plt.plot(x, y_ml, label=f"AI  (f={sag_ml:.3f} m)")
    plt.title(f"Stress-Ribbon Profile – L={L} m")
    plt.xlabel("Length (m)")
    plt.ylabel("Sag (m)")
    ylim = max(sag_eng, sag_ml or 0) * 1.1
    plt.ylim(ylim, -0.1 * ylim)
    plt.grid(":", alpha=.7)
    plt.legend()
    plt.gcf().text(0.995, 0.01, CREDIT_TEXT, ha="right", va="bottom",
                   fontsize=6, style="italic", alpha=.7)
    st.pyplot(plt.gcf())
    plt.close()

    # ------------- 3-D plot -------------------------------------------------
    plot_sag = sag_ml if sag_ml else sag_eng
    if n > 0 and plot_sag > 0:
        fig = plt.figure(figsize=(10, 6))
        ax  = fig.add_subplot(111, projection="3d")
        y_pos = np.linspace(-(n-1)*spacing/2, (n-1)*spacing/2, n) if n > 1 else [0]
        z_crv = y_ml if y_ml is not None else y_eng
        for i, y0 in enumerate(y_pos, start=1):
            ax.plot(x, np.full_like(x, y0), z_crv, label=f"Cable {i}")
        ax.set_title(f"3-D Layout (n={n}, s={spacing} m)")
        ax.set_xlabel("Length (m)")
        ax.set_ylabel("Transverse (m)")
        ax.set_zlabel("Sag (m)")
        ax.set_zlim(plot_sag*1.1, 0)
        ax.view_init(20, -50)
        if n > 1:
            ax.legend(loc="upper left", bbox_to_anchor=(0.85, 1.0))
        fig.text(0.995, 0.01, CREDIT_TEXT, ha="right", va="bottom",
                 fontsize=6, style="italic", alpha=.7)
        st.pyplot(fig)
        plt.close()

    # credit
    st.markdown(f"<span style='font-size:8px;font-style:italic;'>{CREDIT_TEXT}</span>",
                unsafe_allow_html=True)

    # remember inputs / sag for feedback
    st.session_state["last_inputs"] = dict(
        Span=L, UDL=w, **{"No. Cables": n, "Spacing": spacing,
                          "Dia": dia_mm, "Strength": strength})
    st.session_state["last_sag"] = sag_eng


# --------------------------- feedback save ----------------------------------
def save_feedback():
    if "last_inputs" not in st.session_state:
        st.warning("Run a calculation first.")
        return

    row = [st.session_state["last_inputs"].get(k, 0) for k in
           ['Span', 'UDL', 'No. Cables', 'Spacing', 'Dia', 'Strength']]
    row += [round(st.session_state.get("last_sag", 0), 3), rating]

    path = "feedback_log.csv"
    need_header = not os.path.exists(path) or os.path.getsize(path) == 0
    try:
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if need_header:
                writer.writerow(['Span','UDL','No. Cables','Spacing',
                                 'Dia','Strength','Sag','Feedback'])
            writer.writerow(row)
        st.success(f"Feedback {rating}% saved (sag = {row[-2]:.3f} m).")
    except Exception as e:
        st.error(f"Feedback save failed: {e}")

# --------------------------- Button wiring ----------------------------------
if calc_clicked:
    run_calculation()

if feedback_click:
    save_feedback()
