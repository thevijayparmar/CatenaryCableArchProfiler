
# =============
#  This is Duplicate copy of original file just to make it compatible with Streamlit platform
#  Stress-Ribbon / Catenary Cable Profiler
#
#  ⚠️  NOTE TO FUTURE MAINTAINERS: Written at multiple mid-night sessiongs with coffee in hand.
#      If something looks weird, blame the caffeine, not the code.
#
#  ✍️  Authors: Vijaykumar Parmar & Dr. K. B. Parikh  (2025)
# ===============

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import itertools, os, csv, math, warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

CREDIT_TEXT = "© Vijaykumar Parmar & Dr. K. B. Parikh"

global_sag = 0.0
ml_model = None
feedback_data_count = 0
last_inputs = {}

def get_builtin_data():
    spans = [40, 60, 80, 100, 120]
    extended_spans = list(range(100, 1300, 100))
    udls = [10, 15, 20]
    cables = [2, 3, 4]
    dias = [12, 14, 16]
    strength = 1860
    spacing_fixed = 1.5

    base_combos = list(itertools.product(spans, udls, cables, dias))[:30]
    ext_combos = list(itertools.product(extended_spans, udls, cables, dias))[:70]
    all_combos = base_combos + ext_combos

    rows = []
    for L, w, n, d in all_combos:
        area = np.pi * (d / 2) ** 2
        H = (n * area * strength) / 1000
        sag = (w * L ** 2) / (8 * H)
        rows.append([L, w, n, spacing_fixed, d, strength, round(sag, 3), 90])

    cols = ['Span', 'UDL', 'No. Cables', 'Spacing', 'Dia', 'Strength', 'Sag', 'Feedback']
    return pd.DataFrame(rows, columns=cols)

def train_ml_model():
    global feedback_data_count

    try:
        df_builtin = get_builtin_data()
        feedback_data_count = 0
        if os.path.exists('feedback_log.csv'):
            df_fb_raw = pd.read_csv('feedback_log.csv')
            if list(df_fb_raw.columns) == ['Span', 'UDL', 'No. Cables', 'Spacing', 'Dia', 'Strength', 'Sag', 'Feedback']:
                df_fb = df_fb_raw[df_fb_raw['Feedback'] > 60]
                feedback_data_count = len(df_fb)
            else:
                df_fb = pd.DataFrame()
        else:
            df_fb = pd.DataFrame()

        df_train = pd.concat([df_builtin, df_fb], ignore_index=True)
        if len(df_train) < 10:
            return None

        X = df_train[['Span', 'UDL', 'No. Cables', 'Spacing', 'Dia', 'Strength']]
        y = df_train['Sag']
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, min_samples_split=5)
        model.fit(X, y)
        return model

    except Exception as err:
        st.error(f"ML training failed: {err}")
        return None

def main():
    st.title("Stress-Ribbon / Catenary Cable Profiler")

    with st.sidebar:
        st.header("Input Parameters")
        L = st.number_input("Span L (m)", min_value=1.0, value=50.0)
        w = st.number_input("UDL w (kN/m)", value=5.0)
        n = st.number_input("No. of Cables", min_value=1, step=1, value=2)
        spacing = st.number_input("Cable Spacing (m)", value=1.0)
        dia_mm = st.number_input("Cable Dia d (mm)", value=20.0)
        strength = st.number_input("Tensile Strength σ_tu (N/mm²)", value=1600.0)
        mode = st.radio("Mode", ["Mathematical Only", "Mathematical + AI"])

    if st.button("Generate Profile & Calculate"):
        global global_sag, ml_model, last_inputs

        last_inputs = {
            "Span": L, "UDL": w,
            "No. Cables": n, "Spacing": spacing,
            "Dia": dia_mm, "Strength": strength
        }

        if not all(val > 0 for val in [L, n, dia_mm, strength]) or w < 0:
            st.error("All inputs must be positive (UDL may be zero).")
            return

        if n == 1:
            spacing = 0

        area_mm2 = math.pi * (dia_mm / 2) ** 2
        H_kN = n * area_mm2 * strength / 1000
        sag_eng = (w * L ** 2) / (8 * H_kN)
        global_sag = sag_eng

        sag_ratio = sag_eng / L * 100
        V_kN = w * L / 2
        T_kN = math.hypot(H_kN, V_kN)
        σ_actual = T_kN * 1000 / (n * area_mm2)

        sag_ml = None
        if mode == "Mathematical + AI":
            ml_model = train_ml_model()
            if ml_model is not None:
                Xnew = pd.DataFrame([[L, w, n, spacing, dia_mm, strength]],
                                    columns=['Span', 'UDL', 'No. Cables', 'Spacing', 'Dia', 'Strength'])
                try:
                    sag_ml = ml_model.predict(Xnew)[0]
                    if sag_ml <= 0 or sag_ml > L:
                        sag_ml = None
                except Exception as e:
                    st.warning(f"AI prediction failed: {e}")

        st.subheader("Results")
        st.text(f"Cable area (single): {area_mm2:.2f} mm²")
        st.text(f"Horizontal tension H: {H_kN:.2f} kN")
        st.text(f"Vertical reaction V: {V_kN:.2f} kN")
        st.text(f"Math sag f_eng: {sag_eng:.3f} m ({sag_ratio:.2f}% of span)")
        st.text(f"Approx. tension at support: {T_kN:.2f} kN")
        st.text(f"Actual stress σ: {σ_actual:.2f} MPa")
        st.success("σ within limit." if σ_actual <= strength else "⚠️ σ exceeds σ_tu!")

        if sag_ml:
            st.text(f"AI sag f_ml: {sag_ml:.3f} m ({sag_ml / L * 100:.2f}% of span)")

        x = np.linspace(0, L, 100)
        y_eng = (4 * sag_eng / L ** 2) * x * (L - x)
        y_ml = (4 * sag_ml / L ** 2) * x * (L - x) if sag_ml else None

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, y_eng, '--', label=f'Eng (f={sag_eng:.3f} m)')
        if y_ml is not None:
            ax.plot(x, y_ml, label=f'AI (f={sag_ml:.3f} m)')
        ax.set_title(f"Stress-Ribbon Profile – L={L} m")
        ax.set_xlabel("Length (m)")
        ax.set_ylabel("Sag (m)")
        ax.set_ylim(max(sag_eng, sag_ml or 0) * 1.1, -0.1)
        ax.grid(True)
        ax.legend()
        fig.text(0.99, 0.01, CREDIT_TEXT, ha='right', va='bottom', fontsize=6, style='italic', alpha=.7)
        st.pyplot(fig)

        st.caption(CREDIT_TEXT)

    st.markdown("---")
    if st.button("Submit Feedback"):
        feedback = st.slider("Rate Output (%)", 0, 100, 80)
        row = [last_inputs.get(k, 0) for k in
               ['Span', 'UDL', 'No. Cables', 'Spacing', 'Dia', 'Strength']]
        row += [round(global_sag, 3), feedback]

        file_path = 'feedback_log.csv'
        need_header = not os.path.exists(file_path)
        try:
            with open(file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if need_header:
                    writer.writerow(['Span','UDL','No. Cables','Spacing','Dia','Strength','Sag','Feedback'])
                writer.writerow(row)
            st.success(f"✅ Feedback {feedback}% saved (Sag = {global_sag:.3f} m).")
        except Exception as err:
            st.error(f"Feedback save failed: {err}")

if __name__ == "__main__":
    main()
