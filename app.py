import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="CGPA to Package Predictor",
    page_icon="🎓",
    layout="centered"
)

# ---------------- Load Model ----------------
model = joblib.load("regression_model.joblib")

# ---------------- Sidebar ----------------
st.sidebar.title("ℹ️ About App")
st.sidebar.write("""
This app predicts **placement package (LPA)** based on **CGPA**  
using a **Linear Regression model**.
""")

st.sidebar.markdown("**Model Type:** Linear Regression")
st.sidebar.markdown("**Input:** CGPA")
st.sidebar.markdown("**Output:** Package (LPA)")

# ---------------- Main UI ----------------
st.title("🎓 CGPA → Package Prediction")
st.markdown("### Enter your CGPA and see the predicted package 📈")

# Input slider (more interactive than number_input)
cgpa = st.slider(
    "Select CGPA",
    min_value=0.0,
    max_value=10.0,
    value=7.0,
    step=0.1
)

# ---------------- Prediction ----------------
if st.button("🚀 Predict Package"):
    cgpa_array = np.array([[cgpa]])
    prediction = model.predict(cgpa_array)
    predicted_package = float(prediction[0])

    st.success(f"💰 **Predicted Package:** {predicted_package:.2f} LPA")

    # ---------------- Chart ----------------
    st.markdown("### 📊 CGPA vs Package Trend")

    # Create CGPA range for line
    cgpa_range = np.linspace(0, 10, 100).reshape(-1, 1)
    package_range = model.predict(cgpa_range)

    fig, ax = plt.subplots()
    ax.plot(cgpa_range, package_range, label="Regression Line")
    ax.scatter(cgpa, predicted_package, label="Your Prediction", marker="o")
    ax.set_xlabel("CGPA")
    ax.set_ylabel("Package (LPA)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("👨‍💻 Built with **Streamlit + Machine Learning**")
