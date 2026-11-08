"""
train_models.py  -  ML Assignment 2 (BITS WILP M.Tech AIML)

What this script does (run it ONCE on BITS Virtual Lab):
  1. Loads ONE dataset (CSV)                      -> Step 1 of assignment
  2. Splits it into train / test                  -> test part saved as test_data.csv
  3. Builds 5 classifiers (with preprocessing)    -> Step 2
  4. Computes Accuracy, AUC, Precision, Recall,
     F1, MCC for every model                      -> Step 2
  5. Saves trained models (model/*.joblib) and
     a results table (results.csv / results.md)   -> Step 3 + README table

Run:  python train_models.py
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from metrics_utils import compute_metrics   # our own metric function (shared with app.py)

# ----------------------------------------------------------------------------
# CONFIG  -> CHANGE THESE TWO LINES TO USE YOUR OWN KAGGLE / UCI DATASET
# ----------------------------------------------------------------------------
DATA_PATH = "data/dataset.csv"   # full dataset csv (min 12 features, min 500 rows)
TARGET_COL = "target"            # name of the column you want to predict
RANDOM_STATE = 42                # makes results reproducible
TEST_SIZE = 0.20                 # 20% rows are kept aside for testing

os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ----------------------------------------------------------------------------
# STEP 1: LOAD DATA
# ----------------------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    # Fallback demo dataset so the project runs out-of-the-box.
    # (Breast Cancer Wisconsin: 569 rows, 30 features.)
    # !! Replace with your own dataset to avoid same-dataset plagiarism flags !!
    from sklearn.datasets import load_breast_cancer
    demo = load_breast_cancer(as_frame=True).frame
    demo.to_csv(DATA_PATH, index=False)
    print("Demo dataset written to", DATA_PATH)

df = pd.read_csv(DATA_PATH)
print("Dataset shape (rows, columns):", df.shape)

# Sanity check for the assignment rules
n_features = df.shape[1] - 1
assert n_features >= 12, "Assignment needs at least 12 features"
assert df.shape[0] >= 500, "Assignment needs at least 500 rows"

X = df.drop(columns=[TARGET_COL])
y_raw = df[TARGET_COL]

# Models need numbers as labels. LabelEncoder maps e.g. {'No','Yes'} -> {0,1}
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
n_classes = len(label_encoder.classes_)
print("Classes:", list(label_encoder.classes_))

# ----------------------------------------------------------------------------
# STEP 2: TRAIN / TEST SPLIT
# ----------------------------------------------------------------------------
# stratify=y keeps the class ratio the same in train and test (fair evaluation)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Save test data (features + original target labels) -> uploaded later in Streamlit app
test_df = X_test.copy()
test_df[TARGET_COL] = label_encoder.inverse_transform(y_test)
test_df.to_csv("test_data.csv", index=False)
print("Saved test_data.csv with", len(test_df), "rows")

# ----------------------------------------------------------------------------
# STEP 3: PREPROCESSING (inside a Pipeline, so it is saved WITH the model)
# ----------------------------------------------------------------------------
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

numeric_pipe = Pipeline([
    # fill missing numbers with the median (robust against outliers)
    ("impute", SimpleImputer(strategy="median")),
    # StandardScaler:  z = (x - mean) / std   -> every feature gets mean 0, std 1
    # Needed for Logistic Regression, KNN, SVM (distance/gradient based).
    ("scale", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    # One-hot: a column 'Color' with {Red, Blue} -> two 0/1 columns
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols),
])

# ----------------------------------------------------------------------------
# STEP 4: THE FIVE MODELS
# ----------------------------------------------------------------------------
models = {
    # Logistic Regression: p(y=1|x) = 1 / (1 + e^-(w.x + b))   (sigmoid of a linear score)
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),

    # Decision Tree: repeatedly splits data on the feature that gives the biggest
    # drop in Gini impurity  G = 1 - sum(p_k^2).  max_depth limits overfitting.
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),

    # KNN: predicts the majority class among the k closest points
    # (Euclidean distance d = sqrt(sum (xi - yi)^2)).
    "KNN": KNeighborsClassifier(n_neighbors=7),

    # Gaussian Naive Bayes: P(c|x) ∝ P(c) * prod_i N(x_i | mean_ci, var_ci)
    # assumes features are independent given the class.
    "Naive Bayes": GaussianNB(),

    # SVM (RBF kernel): finds the boundary with the maximum margin;
    # K(x,x') = exp(-gamma * ||x - x'||^2).  probability=True gives scores for AUC.
    "Support Vector Machine": SVC(kernel="rbf", C=1.0, probability=True,
                                  random_state=RANDOM_STATE),
}

# ----------------------------------------------------------------------------
# STEP 5: TRAIN, EVALUATE, SAVE
# ----------------------------------------------------------------------------
rows = []
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)                       # learning happens here
    y_pred = pipe.predict(X_test)                    # predicted class
    y_proba = pipe.predict_proba(X_test)             # predicted probabilities

    m = compute_metrics(y_test, y_pred, y_proba)
    rows.append({"ML Model Name": name, **m})

    fname = "model/" + name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipe, fname)
    print(f"{name:25s} -> saved {fname}")

# Save label encoder so the app can decode numbers back to class names
joblib.dump({"label_encoder": label_encoder, "target_col": TARGET_COL},
            "model/meta.joblib")

results = pd.DataFrame(rows).round(4)
results.to_csv("results.csv", index=False)
with open("results.md", "w") as f:
    f.write(results.to_markdown(index=False))   # paste this table in README.md

print("\n=== COMPARISON TABLE ===")
print(results.to_string(index=False))
best = results.sort_values("MCC", ascending=False).iloc[0]
print("\nBest model by MCC:", best["ML Model Name"])
