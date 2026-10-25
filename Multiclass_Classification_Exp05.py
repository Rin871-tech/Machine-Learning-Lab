# -*- coding: utf-8 -*-
"""
Experiment No. 05
Implementation of Logistic Regression for Multiclass Classification

Based on the professor-provided notebook:
- Part 1: manually created student performance dataset (Low / Medium / High)
- Part 2: Iris flower dataset (Setosa / Versicolor / Virginica)

Install:
    python3 -m pip install -r requirements_multiclass.txt

Run:
    streamlit run Multiclass_Classification_Exp05.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="Experiment 05 | Multiclass Classification",
    page_icon="🌸",
    layout="wide",
)

st.markdown(
    """
    <style>
      .main-title {font-size: 2rem; font-weight: 750; margin-bottom: .2rem;}
      .subtitle {color: #64748b; margin-bottom: 1rem;}
      div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.22);
        padding: .8rem; border-radius: .7rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🌸 Experiment No. 05: Multiclass Classification</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Logistic Regression, OvR, OvO, Decision Tree and SVM</div>',
    unsafe_allow_html=True,
)

with st.expander("Aim, objectives and theory", expanded=False):
    st.markdown(
        r"""
**Aim:** To implement multiclass classification using Python and evaluate its performance.

**Objectives**
1. Study multiclass classification.
2. Implement classification using logistic regression and other classifiers.
3. Evaluate performance using accuracy, confusion matrix, and classification report.
4. Visualise classification boundaries for multiple classes.

**Theory**

Multiclass classification predicts a target with more than two categories. Logistic regression can
handle multiple classes using the multinomial (softmax) approach, One-vs-Rest (OvR), or One-vs-One (OvO).

- **OvR:** trains one classifier per class against all other classes.
- **OvO:** trains a classifier for every pair of classes.
- **Accuracy:** proportion of correct predictions.
- **Confusion matrix:** compares actual and predicted class labels.
- **Precision, recall and F1-score:** report performance for each class.
"""
    )

with st.expander("Procedure followed (based on the professor's notebook)", expanded=False):
    st.markdown(
        """
1. Import NumPy, pandas, Matplotlib and scikit-learn.
2. Load either the notebook's student-performance example or the built-in Iris dataset.
3. Separate independent features and the multiclass target.
4. Split data into training and testing sets.
5. Train multinomial logistic regression or compare it with OvR, OvO, Decision Tree and SVM.
6. Predict classes for the test set.
7. Evaluate accuracy, confusion matrix and classification report.
8. Visualise class distributions and a two-feature decision boundary.
9. Enter custom values to predict the class.
"""
    )

st.sidebar.header("⚙️ Experiment settings")
dataset_choice = st.sidebar.selectbox(
    "Dataset",
    ["Iris flower dataset (recommended)", "Student performance example (from notebook)"],
)
model_choice = st.sidebar.selectbox(
    "Classifier",
    [
        "Multinomial Logistic Regression",
        "Logistic Regression (One-vs-Rest)",
        "Logistic Regression (One-vs-One)",
        "Decision Tree",
        "Support Vector Machine (SVM)",
    ],
)
test_percent = st.sidebar.slider("Test set (%)", min_value=10, max_value=40, value=20, step=5)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)

# ---------------------------------------------------------------------
# Dataset options: mirror the two parts of the professor's notebook.
# ---------------------------------------------------------------------
if dataset_choice == "Student performance example (from notebook)":
    df = pd.DataFrame({
        "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Attendance": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
        "Performance": [0, 0, 0, 1, 1, 1, 2, 2, 2, 2],
    })
    feature_names = ["Study_Hours", "Attendance"]
    target_col = "Performance"
    class_names = {0: "Low", 1: "Medium", 2: "High"}
    class_labels = ["Low", "Medium", "High"]
    problem_description = "Predict student performance level from study hours and attendance."
    input_ranges = {
        "Study_Hours": (0.0, 24.0, 5.0, 0.5),
        "Attendance": (0.0, 100.0, 75.0, 1.0),
    }
else:
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    feature_names = list(iris.feature_names)
    target_col = "species"
    class_names = {i: name.title() for i, name in enumerate(iris.target_names)}
    class_labels = [class_names[i] for i in range(3)]
    problem_description = "Classify an Iris flower as Setosa, Versicolor, or Virginica."
    input_ranges = {
        feature: (
            float(df[feature].min()),
            float(df[feature].max()),
            float(df[feature].median()),
            0.1,
        )
        for feature in feature_names
    }

st.sidebar.caption(problem_description)

# Feature controls: Iris uses all four features for primary metrics; the
# student example uses its two original features.
if dataset_choice.startswith("Iris"):
    selected_features = feature_names
else:
    selected_features = feature_names

X = df[selected_features].astype(float)
y = df[target_col].astype(int)

if len(df) < 6 or y.nunique() < 3:
    st.error("This experiment requires enough rows and at least three target classes.")
    st.stop()

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_percent / 100, random_state=int(random_state)
    )
except ValueError as exc:
    st.error(f"Could not split the dataset: {exc}")
    st.stop()

def create_model(name):
    if name == "Multinomial Logistic Regression":
        # For multiclass problems, lbfgs uses multinomial loss where applicable.
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(solver="lbfgs", max_iter=2000, random_state=int(random_state)),
        )
    if name == "Logistic Regression (One-vs-Rest)":
        return make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(LogisticRegression(max_iter=2000, random_state=int(random_state))),
        )
    if name == "Logistic Regression (One-vs-One)":
        return make_pipeline(
            StandardScaler(),
            OneVsOneClassifier(LogisticRegression(max_iter=2000, random_state=int(random_state))),
        )
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=5, random_state=int(random_state))
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=int(random_state)))

model = create_model(model_choice)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
labels = sorted(y.unique().tolist())
label_names = [class_names[int(label)] for label in labels]
cm = confusion_matrix(y_test, y_pred, labels=labels)
report = classification_report(
    y_test, y_pred, labels=labels, target_names=label_names,
    output_dict=True, zero_division=0
)

# ---------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------
st.subheader("1. Dataset overview")
a, b, c, d = st.columns(4)
a.metric("Dataset rows", f"{len(df):,}")
b.metric("Features", str(len(selected_features)))
c.metric("Training samples", f"{len(X_train):,}")
d.metric("Testing samples", f"{len(X_test):,}")

with st.expander("View dataset and descriptive statistics", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    st.write("Dataset shape:", df.shape)
    st.write("Descriptive statistics")
    st.dataframe(df.describe().T, use_container_width=True)
    st.write("Class counts")
    st.dataframe(
        df[target_col].map(class_names).value_counts().rename_axis("Class").to_frame("Count"),
        use_container_width=True,
    )

st.subheader("2. Classification performance")
k1, k2 = st.columns(2)
k1.metric("Test accuracy", f"{accuracy:.1%}")
k2.metric("Classifier", model_choice)

st.caption(
    f"Train/test split: {100-test_percent}% / {test_percent}%; random state: {int(random_state)}. "
    "Results may vary when these settings change."
)

st.subheader("3. Confusion matrix")
cm_fig, cm_ax = plt.subplots(figsize=(7, 5))
im = cm_ax.imshow(cm, interpolation="nearest")
cm_ax.set_title(f"Confusion Matrix — {model_choice}")
cm_ax.set_xlabel("Predicted class")
cm_ax.set_ylabel("Actual class")
cm_ax.set_xticks(range(len(label_names)), labels=label_names, rotation=20, ha="right")
cm_ax.set_yticks(range(len(label_names)), labels=label_names)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        cm_ax.text(j, i, str(cm[i, j]), ha="center", va="center")
cm_fig.colorbar(im, ax=cm_ax, fraction=0.046, pad=0.04)
cm_fig.tight_layout()
st.pyplot(cm_fig)
plt.close(cm_fig)

st.subheader("4. Classification report")
report_df = pd.DataFrame(report).T
st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

st.subheader("5. Visualise class distribution")
if dataset_choice.startswith("Iris"):
    default_x = "sepal length (cm)"
    default_y = "petal length (cm)"
else:
    default_x = "Study_Hours"
    default_y = "Attendance"

plot_left, plot_right = st.columns(2)
with plot_left:
    x_feature = st.selectbox("X-axis feature", feature_names, index=feature_names.index(default_x))
with plot_right:
    y_options = [f for f in feature_names if f != x_feature]
    preferred_y = default_y if default_y in y_options else y_options[0]
    y_feature = st.selectbox("Y-axis feature", y_options, index=y_options.index(preferred_y))

scatter_fig, scatter_ax = plt.subplots(figsize=(8, 5))
for class_id in labels:
    subset = df[df[target_col] == class_id]
    scatter_ax.scatter(
        subset[x_feature], subset[y_feature],
        label=class_names[int(class_id)], alpha=0.85, edgecolors="black", linewidths=0.35
    )
scatter_ax.set_xlabel(x_feature)
scatter_ax.set_ylabel(y_feature)
scatter_ax.set_title("Multiclass Data Distribution")
scatter_ax.grid(True, alpha=0.2)
scatter_ax.legend(title="Class")
scatter_fig.tight_layout()
st.pyplot(scatter_fig)
plt.close(scatter_fig)

st.subheader("6. Two-feature decision boundary")
st.caption(
    "For this visualisation only, the selected classifier is refitted using the two displayed features. "
    "The main evaluation above uses the full feature set."
)
X_2d = df[[x_feature, y_feature]].astype(float)
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X_2d, y, test_size=test_percent / 100, random_state=int(random_state)
)
boundary_model = create_model(model_choice)
boundary_model.fit(X2_train, y2_train)

x_min, x_max = float(X_2d[x_feature].min()), float(X_2d[x_feature].max())
y_min, y_max = float(X_2d[y_feature].min()), float(X_2d[y_feature].max())
x_pad = max((x_max - x_min) * 0.08, 0.1)
y_pad = max((y_max - y_min) * 0.08, 0.1)
xx, yy = np.meshgrid(
    np.linspace(x_min - x_pad, x_max + x_pad, 240),
    np.linspace(y_min - y_pad, y_max + y_pad, 240),
)
grid = pd.DataFrame({x_feature: xx.ravel(), y_feature: yy.ravel()})
zz = boundary_model.predict(grid).reshape(xx.shape)

boundary_fig, boundary_ax = plt.subplots(figsize=(9, 6))
boundary_ax.contourf(xx, yy, zz, alpha=0.22, levels=np.arange(min(labels)-0.5, max(labels)+1.5, 1))
for class_id in labels:
    subset = df[df[target_col] == class_id]
    boundary_ax.scatter(
        subset[x_feature], subset[y_feature],
        label=class_names[int(class_id)], edgecolors="black", linewidths=0.35, alpha=0.9
    )
boundary_ax.set_xlabel(x_feature)
boundary_ax.set_ylabel(y_feature)
boundary_ax.set_title(f"Decision Boundary — {model_choice}")
boundary_ax.legend(title="Class")
boundary_ax.grid(True, alpha=0.15)
boundary_fig.tight_layout()
st.pyplot(boundary_fig)
plt.close(boundary_fig)

st.subheader("7. Predict a class for new input")
st.write("Enter feature values to predict the corresponding class.")
with st.form("prediction_form"):
    entered = {}
    col_left, col_right = st.columns(2)
    for i, feature in enumerate(feature_names):
        col = col_left if i % 2 == 0 else col_right
        low, high, default, step = input_ranges[feature]
        with col:
            entered[feature] = st.number_input(
                feature.replace("_", " ").title(),
                min_value=float(low),
                max_value=float(high),
                value=float(default),
                step=float(step),
                format="%.2f",
            )
    submitted = st.form_submit_button("Predict class", type="primary")

if submitted:
    input_df = pd.DataFrame([entered], columns=feature_names)
    predicted = int(model.predict(input_df)[0])
    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(input_df)[0]
        except Exception:
            probabilities = None
    st.success(f"Predicted class: **{class_names[predicted]}**")
    if probabilities is not None:
        proba_df = pd.DataFrame({
            "Class": [class_names[int(c)] for c in model.classes_],
            "Probability": probabilities,
        }).sort_values("Probability", ascending=False)
        st.dataframe(
            proba_df.style.format({"Probability": "{:.1%}"}),
            use_container_width=True,
            hide_index=True,
        )

with st.expander("View test-set predictions", expanded=False):
    pred_df = X_test.copy()
    pred_df["Actual class"] = y_test.map(class_names)
    pred_df["Predicted class"] = pd.Series(y_pred, index=y_test.index).map(class_names)
    pred_df["Correct"] = pred_df["Actual class"] == pred_df["Predicted class"]
    st.dataframe(pred_df, use_container_width=True)
    st.download_button(
        "Download test predictions as CSV",
        data=pred_df.to_csv(index=True).encode("utf-8"),
        file_name="multiclass_test_predictions.csv",
        mime="text/csv",
    )

st.subheader("8. Observations and conclusion")
st.markdown(
    f"""
**Observations**
- The selected dataset contains **{len(df)} records** and **{len(labels)} classes**.
- The model assigns each test sample to one of the available classes.
- The confusion matrix shows correct and incorrect predictions for every class.
- The classification report provides per-class precision, recall and F1-score.
- The decision-boundary visualisation shows how the selected two features separate classes.

**Conclusion**

Multiclass classification was implemented using **{model_choice}**. The model was evaluated
using accuracy, confusion matrix and classification report. The class-distribution plot and
two-feature decision boundary help visualise how samples are separated into multiple classes.
"""
)

st.caption("Machine Learning Lab Manual by Dr. Sachin Balawant Takmare | Experiment No. 05")
