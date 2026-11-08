"""
app.py - Streamlit app for ML Assignment 2.

Required features (from assignment):
  a. CSV upload (test data only)
  b. Model selection dropdown
  c. Evaluation metrics display
  d. Confusion matrix + classification report
The app only LOADS saved models - it never retrains.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from metrics_utils import compute_metrics

st.set_page_config(page_title="Classifier Lab", page_icon="🧪", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}


@st.cache_resource            # load each file once, reuse on every rerun (faster)
def load_model(path):
    return joblib.load(path)


meta = load_model("model/meta.joblib")
label_encoder = meta["label_encoder"]
target_col = meta["target_col"]

st.title("🧪 Classifier Lab - compare 5 saved ML models")
st.caption("Upload test data, pick a model, see how it performs. "
           "Models are pre-trained: nothing is retrained here.")

# ---------------- Sidebar: feature (a) upload + feature (b) dropdown -----------
with st.sidebar:
    st.header("1. Upload test data")
    uploaded = st.file_uploader("CSV with features + target column", type="csv")
    st.header("2. Choose model")
    model_name = st.selectbox("Model", list(MODEL_FILES.keys()))
    st.info(f"Target column expected: **{target_col}**")

if uploaded is None:
    st.warning("👈 Upload test_data.csv from the sidebar to begin.")
    st.stop()

data = pd.read_csv(uploaded)
if target_col not in data.columns:
    st.error(f"Column '{target_col}' not found in the uploaded file.")
    st.stop()

X_new = data.drop(columns=[target_col])
# Turn text labels back into the numbers the models were trained on
try:
    y_true = label_encoder.transform(data[target_col])
except ValueError:
    st.error("Target column has labels the model has never seen.")
    st.stop()

with st.expander("Preview uploaded data", expanded=False):
    st.write(f"{data.shape[0]} rows x {data.shape[1]} columns")
    st.dataframe(data.head(20))


def evaluate(name):
    model = load_model(MODEL_FILES[name])
    y_pred = model.predict(X_new)
    y_proba = model.predict_proba(X_new)
    return y_pred, compute_metrics(y_true, y_pred, y_proba)


tab1, tab2, tab3 = st.tabs(["📊 Selected model", "🏁 All models", "🔍 Predictions"])

# ---------------- Feature (c) metrics + (d) confusion matrix/report -----------
with tab1:
    y_pred, m = evaluate(model_name)
    st.subheader(model_name)
    cols = st.columns(6)
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**  (rows = actual, columns = predicted)")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.markdown("**Classification report**")
        rep = classification_report(y_true, y_pred, output_dict=True,
                                    target_names=[str(c) for c in label_encoder.classes_],
                                    zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3))

with tab2:
    st.subheader("All five models on the uploaded data")
    table = pd.DataFrame([{"ML Model Name": n, **evaluate(n)[1]}
                          for n in MODEL_FILES]).set_index("ML Model Name").round(4)
    st.dataframe(table.style.highlight_max(axis=0, color="#c6efce"))
    st.bar_chart(table[["Accuracy", "F1", "MCC"]])
    st.caption(f"Best MCC: **{table['MCC'].idxmax()}**")

with tab3:
    out = data.copy()
    out["predicted"] = label_encoder.inverse_transform(y_pred)
    out["correct"] = out["predicted"] == out[target_col]
    st.dataframe(out)
    st.download_button("Download predictions", out.to_csv(index=False),
                       "predictions.csv", "text/csv")
