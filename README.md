
# Stress-Ribbon / Catenary Cable Profiler

An interactive Python tool for estimating cable sag profiles in **Stress Ribbon Bridges** (SRBs) using engineering equations and machine learning (Random Forest). Designed for researchers, bridge engineers, and students interested in early-stage SRB design.

---

## 🔍 Features

- 📐 **Mathematical Mode**: Computes cable sag using classical parabolic formulas.
- 🤖 **AI Mode**: Uses a Random Forest model trained on synthetic data and user feedback to adjust sag predictions.
- 📊 **2D and 3D Visualizations**: View the cable profile in a parabolic arc (2D) and spatially (3D).
- 🧠 **Learns from You**: You can rate the results, and the model will improve with each feedback.

---

## 📁 File Structure

```
.
├── app.py                  # Main Streamlit app (you must rename your working file to app.py)
├── feedback_log.csv        # Stores user feedback (created during use)
├── requirements.txt        # Required libraries
└── README.md               # This file
```

---

## 🚀 Run the Tool

### Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```

### On [Streamlit Cloud](https://streamlit.io/cloud)
1. Upload all files to a GitHub repository.
2. Sign in at [streamlit.io/cloud](https://streamlit.io/cloud).
3. Create a new app pointing to your repo and `app.py`.

---

## ⚙️ Input Parameters

| Parameter         | Unit       | Description                                 |
|------------------|------------|---------------------------------------------|
| Span (L)         | meters     | Total length between cable supports         |
| UDL (w)          | kN/m       | Uniformly distributed load                  |
| No. of Cables (n)| count      | Number of cables sharing the load           |
| Cable Spacing    | meters     | Distance between cables                     |
| Cable Diameter   | mm         | Diameter of one cable                       |
| Tensile Strength | MPa (N/mm²)| Allowable or ultimate tensile strength      |

---

## 💡 How It Works

1. **Math Engine**: Uses formula `f = (w * L²) / (8 * H)` to calculate sag.
2. **AI Mode**:
   - Trained on ~100 synthetic combinations.
   - Enhanced with feedback above 60% rating.
   - Automatically retrains during sessions if new feedback is added.

---

## ✍️ Feedback Logging

If you're satisfied with the result, use the **slider** to rate it and press **Submit Feedback**. Your input will be stored and used in future AI model training.

---

## 📜 License & Credits

Developed by:
- Vijaykumar Parmar
- Dr. K. B. Parikh 

This tool is free to use and intended for academic and research purposes.

---

## 🛠 Planned Extensions (Future Work)

- Support for multi-span stress ribbon analysis
- Real-time sensor integration
- True catenary calculation module
- Deck stiffness interaction modeling

---

For questions or collaboration, feel free to raise an issue or contact the authors.
