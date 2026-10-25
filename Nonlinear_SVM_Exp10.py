import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

st.set_page_config(page_title="Experiment 10 | Non-Linear SVM", page_icon="🌀", layout="wide")
st.title("Experiment No. 10 — Non-Linear Support Vector Machine")
st.caption("Compare Polynomial and RBF kernels on non-linearly separable data.")

with st.sidebar:
    st.header("Dataset")
    dataset_name = st.selectbox("Choose dataset", ["make_moons", "make_circles"])
    noise = st.slider("Noise", 0.0, 0.45, 0.20, 0.05)
    n_samples = st.slider("Number of samples", 200, 1500, 600, 100)
    st.header("SVM settings")
    test_size = st.slider("Test set (%)", 10, 40, 25, 5) / 100
    c_value = st.select_slider("C (error penalty)", options=[0.1, 1.0, 10.0, 100.0], value=1.0)
    gamma = st.selectbox("Gamma", ["scale", "auto", 0.1, 1.0, 5.0], index=0)
    degree = st.slider("Polynomial degree", 2, 6, 3)

if dataset_name == "make_moons":
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
else:
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.45, random_state=42)
df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
df["Class"] = y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

st.subheader("Dataset")
fig, ax = plt.subplots()
for label in np.unique(y):
    ax.scatter(X[y == label, 0], X[y == label, 1], s=20, alpha=0.75, label=f"Class {label}")
ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2"); ax.legend()
st.pyplot(fig)

models = {
    "Polynomial": make_pipeline(StandardScaler(), SVC(kernel="poly", degree=degree, C=c_value, gamma=gamma)),
    "RBF": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=c_value, gamma=gamma))
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = (model, pred, accuracy_score(y_test, pred))

st.subheader("Kernel comparison")
cols = st.columns(2)
for col, (name, (model, pred, acc)) in zip(cols, results.items()):
    with col:
        st.metric(f"{name} test accuracy", f"{acc:.3%}")
        st.write(f"Support vectors: **{int(model[-1].n_support_.sum())}**")
        st.code(classification_report(y_test, pred, zero_division=0), language="text")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot(ax=ax, colorbar=False)
        ax.set_title(f"{name} confusion matrix")
        st.pyplot(fig)

st.subheader("Decision boundaries and support vectors")
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 250),
                     np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 250))
grid = np.c_[xx.ravel(), yy.ravel()]
cols = st.columns(2)
for col, (name, (model, pred, acc)) in zip(cols, results.items()):
    with col:
        zz = model.predict(grid).reshape(xx.shape)
        fig, ax = plt.subplots()
        ax.contourf(xx, yy, zz, alpha=0.22, levels=[-0.5, 0.5, 1.5])
        ax.scatter(X_train[:,0], X_train[:,1], c=y_train, s=18, edgecolors="k", linewidths=0.2)
        sv = model[-1].support_vectors_
        ax.scatter(sv[:,0], sv[:,1], s=75, facecolors="none", edgecolors="red", linewidths=1.2, label="Support vectors")
        ax.set_title(f"{name} kernel boundary")
        ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2"); ax.legend()
        st.pyplot(fig)

st.subheader("Generated dataset preview")
st.dataframe(df.head(15), use_container_width=True)
st.caption("Pipeline: StandardScaler → SVC(kernel='poly' or 'rbf'). Compare accuracy, confusion matrices, classification reports, and decision boundaries.")
