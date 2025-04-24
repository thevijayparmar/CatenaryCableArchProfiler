# =============
#  Stress-Ribbon  Catenary Cable Profiler
#
#  ⚠️  NOTE TO FUTURE MAINTAINERS Written at multiple mid-night sessiongs with coffee in hand.
#      If something looks weird, blame the coffee, not the code.
#
#  ✍️  Authors Vijaykumar Parmar & Dr. K. B. Parikh  (2025)
# ===============

# 1. Imports (Unused imports removed)
import streamlit as st
import numpy as np
import pandas as pd
import csv, os, warnings, itertools, math   # math not needed but handy
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa F401  (used implicitly)

# “For… no noisy warnings while demo-ing to the committee.”
warnings.filterwarnings('ignore', category=UserWarning,  module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

CREDIT_TEXT = © Vijaykumar Parmar & Dr. K. B. Parikh

# 2. Widgets (To make it look good to easy to operate)
input_col1 = widgets.VBox([
    widgets.FloatText(value=50.0, description='Span L (m)'),
    widgets.FloatText(value=5.0,  description='UDL w (kNm)'),
    widgets.IntText(  value=2,    description='No. of Cables n')
])

input_col2 = widgets.VBox([
    widgets.FloatText(value=1.0,   description='Cable Spacing (m)'),
    widgets.FloatText(value=20.0,  description='Cable Dia d (mm)'),
    widgets.FloatText(value=1600., description='Tensile σ_tu (Nmm²)')
])

# Design factor (user‑set – defaults to 0.6)
design_factor_widget = widgets.FloatText(value=0.6, description='Design Factor')

# Option for selection
mode_selector = widgets.RadioButtons(
    options=['Mathematical Only', 'Mathematical + AI'],
    value='Mathematical Only',
    description='Mode',
    layout={'width' 'max-content'}
)

calc_button     = widgets.Button(description='Generate Profile & Calculate',
                                 button_style='info')
feedback_slider = widgets.IntSlider(80, 0, 100,
                                    description='Rate Output (%)',
                                    style={'description_width' 'initial'})
submit_button   = widgets.Button(description='Submit Feedback',
                                 button_style='success')

output_area = widgets.Output()

# Display widgets
display(widgets.HBox([input_col1, input_col2, design_factor_widget]),
        widgets.VBox([mode_selector, calc_button]),
        widgets.HBox([feedback_slider, submit_button]),
        output_area)

# 3. Globals ( but convenient for a quick notebook demo) ----
global_sag          = 0.0
ml_model            = None
feedback_data_count = 0
last_inputs         = {}

# 4. Synthetic Data Generator (for Aabra ka daabra - So that AI can work at Zero feedbacks also✨) --
def get_builtin_data()
    
    Use up ~100 synthetic training points so that the RF model doesn’t starve.
    
    spans           = [40, 60, 80, 100, 120]
    extended_spans  = list(range(100, 1300, 100))   # up to 1200 m
    udls            = [10, 15, 20]
    cables          = [2, 3, 4]
    dias            = [12, 14, 16]
    strength        = 1860                           # MPa (fixed)

    spacing_fixed   = 1.5                            # m

    base_combos = list(itertools.product(spans, udls, cables, dias))[30]
    ext_combos  = list(itertools.product(extended_spans, udls, cables, dias))[70]
    all_combos  = base_combos + ext_combos

    rows = []
    for L, w, n, d in all_combos
        area = np.pi  (d  2)  2
        H    = (n  area  strength)  1000          # kN
        sag  = (w  L  2)  (8  H)
        rows.append([L, w, n, spacing_fixed, d, strength,
                     round(sag, 3), 90])             # optimistic 90 % rating

    cols = ['Span', 'UDL', 'No. Cables', 'Spacing',
            'Dia', 'Strength', 'Sag', 'Feedback']
    return pd.DataFrame(rows, columns=cols)

# -- 5. Model Training --
def train_ml_model()
    
    Fit a RandomForest on synthetic + user-feedback rows (rating  60 %).
    Returns the trained model or None if training fizzles.
    
    global feedback_data_count

    try
        df_builtin = get_builtin_data()
        print(fℹ️  Built-in rows {len(df_builtin)})

        # --- Load user feedback, if any -------------------------------
        feedback_data_count = 0
        if os.path.exists('feedback_log.csv')
            df_fb_raw = pd.read_csv('feedback_log.csv')
            expected_cols = ['Span','UDL','No. Cables','Spacing',
                             'Dia','Strength','Sag','Feedback']
            if list(df_fb_raw.columns) == expected_cols
                df_fb = df_fb_raw[df_fb_raw['Feedback']  60]
                feedback_data_count = len(df_fb)
                if feedback_data_count
                    print(fℹ️  Feedback rows {feedback_data_count})
            else
                print(⚠️  Feedback CSV column mismatch—ignored.)
        else
            df_fb = pd.DataFrame()

        df_train = pd.concat([df_builtin, df_fb], ignore_index=True)
        if len(df_train)  10               # unlikely, but be safe
            print(⚠️  Too little data ( 10) for ML training.)
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

        print(f✅  RF model trained on {len(df_train)} rows.)
        return model

    except Exception as err
        print(f❌  ML training failed {err})
        return None

# 6. Core Calc + Plot (works when clicked on button) --
def calculate_and_plot(_unused_event)
    global global_sag, ml_model, feedback_data_count, last_inputs

    with output_area
        clear_output(wait=True)
        print(Crunching numbers...)

        # ---- Gather inputs -------------------------------------------------
        L, w, n = (child.value for child in input_col1.children)
        spacing, dia_mm, strength = (child.value for child in input_col2.children)
        design_factor = design_factor_widget.value

        last_inputs = dict(
            Span=L, UDL=w,
            {'No. Cables' n, 'Spacing' spacing, 'Dia' dia_mm, 'Strength' strength}
        )

        if not all(val  0 for val in [L, n, dia_mm, strength, design_factor]) or w  0 or not (0  design_factor = 1)
            print(❌  All inputs must be positive (UDL may be zero) and 0  Design Factor ≤ 1.)
            return

        if n == 1   # spacing meaningless for a single cable
            if spacing != 0
                print(ℹ️  n = 1 ⇒ spacing set to 0.)
                spacing = 0
                input_col2.children[0].value = 0

        # ---- Engineering maths (plain vanilla) ---------------------------
        area_mm2   = math.pi  (dia_mm  2)  2
        H_kN       = n  area_mm2  (design_factor  strength)  1000  # adjusted horizontal force
        sag_eng    = (w  L  2)  (8  H_kN)
        global_sag = sag_eng

        sag_ratio  = sag_eng  L  100
        V_kN       = w  L  2
        T_kN       = math.hypot(H_kN, V_kN)
        σ_actual   = T_kN  1000  (n  area_mm2)
        utilization = (σ_actual  strength)  100

        # ---- Optional AI adjustment --------------------------------------
        sag_ml = sag_ratio_ml = None
        if mode_selector.value == 'Mathematical + AI'
            print(nTraining  loading ML model...)
            ml_model = train_ml_model()
            if ml_model is not None
                Xnew = pd.DataFrame([[L, w, n, spacing, dia_mm, strength]],
                                    columns=['Span','UDL','No. Cables',
                                             'Spacing','Dia','Strength'])
                try
                    sag_ml = ml_model.predict(Xnew)[0]
                    if sag_ml = 0 or sag_ml  L
                        print(f⚠️  AI sag {sag_ml.2f} m is fishy. Ignored.)
                        sag_ml = None
                    else
                        sag_ratio_ml = sag_ml  L  100
                except Exception as e
                    print(f❌  AI prediction failed {e})

        # ---- Printout -----------------------------------------------------
        print(n--- RESULTS ---)
        print(fL = {L} m  w = {w} kNm  n = {n}  d = {dia_mm} mm 
              f σ_tu = {strength} MPa  s = {spacing} m  ϕ = {design_factor})
        print(fCable area (single)      {area_mm28.2f} mm²)
        print(fHorizontal tension H     {H_kN8.2f} kN)
        print(fVertical reaction V      {V_kN8.2f} kN)
        print(-  32)
        print(f🧮 Math sag f_eng        {sag_eng8.3f} m 
              f({sag_ratio.2f} % of span))
        print(fApprox. tension at supp  {T_kN8.2f} kN)
        print(fActual stress σ          {σ_actual8.2f} MPa)
        print(fUtilization              {utilization6.1f} % of σ_tu)
        print(⚠️  σ exceeds σ_tu! if σ_actual  strength
              else ✅  σ within limit.)

        if sag_ml is not None
            print(-  32)
            print(f🤖 AI sag f_ml           {sag_ml8.3f} m 
                  f({sag_ratio_ml.2f} % of span))
            print(f(trained on {feedback_data_count} feedback rows))

        # ---- 2-D Plot -----------------------------------------------------
        x = np.linspace(0, L, 100)
        y_eng = (4  sag_eng  L  2)  x  (L - x)
        y_ml  = (4  sag_ml  L  2)  x  (L - x) if sag_ml else None

        plt.figure(figsize=(10, 4))
        plt.plot(x, y_eng, '--', label=f'Eng (f={sag_eng.3f} m)')
        if y_ml is not None
            plt.plot(x, y_ml, label=f'AI  (f={sag_ml.3f} m)')
        plt.title(fStress-Ribbon Profile – L={L} m)
        plt.xlabel(Length (m))
        plt.ylabel(Sag (m))
        ylim = max(sag_eng, sag_ml or 0)  1.1
        plt.ylim(ylim, -0.1  ylim)
        plt.grid('', alpha=.7)
        plt.legend()
        plt.gcf().text(0.99, 0.01, CREDIT_TEXT,
                       ha='right', va='bottom',
                       fontsize=6, style='italic', alpha=.7)
        plt.show()

        # - 3-D Plot (because it gives ❤ eye-candy) -
        plot_sag = sag_ml if sag_ml else sag_eng
        if n  0 and plot_sag  0
            fig = plt.figure(figsize=(10, 6))
            ax  = fig.add_subplot(111, projection='3d')

            y_pos = (np.linspace(-(n - 1)  spacing  2,
                                 (n - 1)  spacing  2, n) if n  1 else [0])
            z_crv = y_ml if y_ml is not None else y_eng
            for i, y0 in enumerate(y_pos, start=1)
                ax.plot(x, np.full_like(x, y0), z_crv, label=f'Cable {i}')
            ax.set_title(f3-D Layout (n = {n}, s = {spacing} m))
            ax.set_xlabel(Length (m))
            ax.set_ylabel(Transverse (m))
            ax.set_zlabel(Sag (m))
            ax.set_zlim(plot_sag  1.1, 0)
            ax.view_init(20, -50)
            if n  1
                ax.legend(loc='upper left', bbox_to_anchor=(0.85, 1.0))
            fig.text(0.99, 0.01, CREDIT_TEXT,
                     ha='right', va='bottom',
                     fontsize=6, style='italic', alpha=.7)
            plt.tight_layout()
            plt.show()

        # ---- Tiny credit (HTML) ------------------------------------------
        display(HTML(fspan style='font-size8px;font-styleitalic;'{CREDIT_TEXT}span))

# -- 7. Feedback Saver --------------------------------------------------------

def save_feedback(_unused_event)
    
    Append the latest calc + user rating to feedback_log.csv.
    
    global last_inputs, global_sag

    with output_area
        if not last_inputs
            print(❌  Run a calculation before saving feedback.)
            return

        feedback_value = feedback_slider.value
        row = [last_inputs.get(k, 0) for k in
               ['Span', 'UDL', 'No. Cables', 'Spacing', 'Dia', 'Strength']]
        row += [round(global_sag, 3), feedback_value]

        file_path   = 'feedback_log.csv'
        need_header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
        try
            with open(file_path, 'a', newline='') as f
                writer = csv.writer(f)
                if need_header
                    writer.writerow(['Span','UDL','No. Cables','Spacing',
                                     'Dia','Strength','Sag','Feedback'])
                writer.writerow(row)
            print(f✅  Feedback {feedback_value} % saved 
                  f(Sag = {global_sag.3f} m).)
        except Exception as err
            print(f❌  Feedback save failed {err})

# -- 8. Wire up buttons (simple & sweet) --------------------------------------
calc_button.on_click(calculate_and_plot)
submit_button.on_click(save_feedback)

# (Uncomment to pre-train once on load—handy for offline demos)
# print(Training model on start-up…)
# ml_model = train_ml_model()
