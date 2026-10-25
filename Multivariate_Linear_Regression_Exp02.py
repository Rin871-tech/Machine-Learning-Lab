# -*- coding: utf-8 -*-
"""
Experiment: Implementation of Multivariate Linear Regression
Dataset: Cgpa_iq_package.csv
Features: cgpa, iq
Target: package (LPA)

Run:
    python -m pip install -r requirements.txt
    python -m streamlit run Multivariate_Linear_Regression_Exp02.py
"""

from pathlib import Path
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Experiment 2 | Multivariate Linear Regression",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Experiment: Implementation of Multivariate Linear Regression")
st.caption("Machine Learning Lab Manual • Interactive experiment interface")

with st.expander("Aim, objectives and theory", expanded=False):
    st.markdown(
        """
**Aim:** To implement multivariate linear regression using Python and evaluate its performance.

**Objectives**
1. Study the concept of multivariate linear regression.
2. Compute model parameters using the closed-form solution and gradient descent.
3. Evaluate regression performance using MSE and R².
4. Visualise regression results with multiple features.

**Theory**

Multivariate linear regression models a target \(y\) from multiple independent variables:

\[
y = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b + \\varepsilon
\]

For this dataset, \(x_1\) is CGPA, \(x_2\) is IQ, and \(y\) is placement package in LPA.

- **Closed-form least squares:** estimates the coefficients by solving the least-squares problem.
- **Gradient descent:** iteratively updates coefficients to minimise mean squared error.
- **MSE:** average squared prediction error; lower is better.
- **R²:** proportion of target variance explained by the model; closer to 1 is generally better.
"""
    )

st.sidebar.header("Experiment settings")
test_size = st.sidebar.slider(
    "Test set proportion", min_value=0.10, max_value=0.40, value=0.20, step=0.05
)
random_state = st.sidebar.number_input(
    "Random state", min_value=0, max_value=9999, value=42, step=1
)
epochs = st.sidebar.slider(
    "Gradient descent epochs", min_value=100, max_value=10000, value=4000, step=100
)
learning_rate = st.sidebar.select_slider(
    "Gradient descent learning rate",
    options=[0.001, 0.003, 0.005, 0.01, 0.03, 0.05],
    value=0.03,
)
uploaded_file = st.sidebar.file_uploader(
    "Upload Cgpa_iq_package.csv (optional)", type=["csv"]
)

@st.cache_data
def read_dataset(file_bytes=None, file_path=None):
    if file_bytes is not None:
        return pd.read_csv(BytesIO(file_bytes))
    if file_path and Path(file_path).exists():
        return pd.read_csv(file_path)
    return None

candidate_paths = [
    Path(__file__).resolve().parent / "Cgpa_iq_package.csv",
    Path.cwd() / "Cgpa_iq_package.csv",
]
default_path = next((p for p in candidate_paths if p.exists()), None)

try:
    if uploaded_file is not None:
        df = read_dataset(file_bytes=uploaded_file.getvalue())
        dataset_name = uploaded_file.name
    elif default_path is not None:
        df = read_dataset(file_path=str(default_path))
        dataset_name = default_path.name
    else:
        df = None
        dataset_name = ""
except Exception as exc:
    df = None
    st.error(f"Could not read the dataset: {exc}")

if df is None:
    st.warning("Upload the professor-provided Cgpa_iq_package.csv using the sidebar.")
    st.stop()

df.columns = [str(c).strip().lower() for c in df.columns]
required = ["cgpa", "iq", "package"]
if not all(col in df.columns for col in required):
    st.error(
        "The dataset must contain columns named cgpa, iq, and package. "
        f"Columns found: {', '.join(df.columns)}"
    )
    st.stop()

raw_rows = len(df)
data = df[required].copy()
for column in required:
    data[column] = pd.to_numeric(data[column], errors="coerce")
data = data.replace([np.inf, -np.inf], np.nan).dropna()
if len(data) < 5:
    st.error("At least five valid numeric rows are required.")
    st.stop()

st.success(f"Dataset loaded: **{dataset_name}**")
if len(data) != raw_rows:
    st.info(f"Removed {raw_rows - len(data)} row(s) with missing or invalid values.")

# Dataset overview
c1, c2, c3, c4 = st.columns(4)
c1.metric("Valid records", f"{len(data)}")
c2.metric("Mean CGPA", f"{data['cgpa'].mean():.2f}")
c3.metric("Mean IQ", f"{data['iq'].mean():.2f}")
c4.metric("Mean package", f"{data['package'].mean():.2f} LPA")

with st.expander("View dataset and descriptive statistics"):
    left, right = st.columns(2)
    left.dataframe(data.head(10), use_container_width=True)
    right.dataframe(data.describe().T, use_container_width=True)

# Features and target match the professor's notebook.
X = data[["cgpa", "iq"]].to_numpy(dtype=float)
y = data["package"].to_numpy(dtype=float)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=float(test_size), random_state=int(random_state)
)

# Model 1: closed-form least-squares solution.
# np.linalg.lstsq is numerically safer than explicitly calculating an inverse.
design_train = np.column_stack([np.ones(len(X_train)), X_train])
theta, *_ = np.linalg.lstsq(design_train, y_train, rcond=None)
intercept_closed = float(theta[0])
coef_closed = theta[1:].astype(float)
pred_closed = intercept_closed + X_test @ coef_closed

# Model 2: gradient descent. Standardise X using training data only.
x_mean = X_train.mean(axis=0)
x_scale = X_train.std(axis=0)
x_scale[x_scale == 0] = 1.0
X_train_scaled = (X_train - x_mean) / x_scale
X_test_scaled = (X_test - x_mean) / x_scale

# Keep target in original units; standardised inputs make the updates stable.
weights_scaled = np.zeros(X_train.shape[1], dtype=float)
bias_scaled = float(y_train.mean())
n = len(y_train)
loss_history = []

for _ in range(int(epochs)):
    y_hat = X_train_scaled @ weights_scaled + bias_scaled
    error = y_hat - y_train
    grad_w = (2.0 / n) * (X_train_scaled.T @ error)
    grad_b = (2.0 / n) * np.sum(error)
    weights_scaled -= float(learning_rate) * grad_w
    bias_scaled -= float(learning_rate) * grad_b
    current_loss = np.mean((X_train_scaled @ weights_scaled + bias_scaled - y_train) ** 2)
    loss_history.append(float(current_loss))

coef_gd = weights_scaled / x_scale
intercept_gd = float(bias_scaled - np.dot(coef_gd, x_mean))
pred_gd = intercept_gd + X_test @ coef_gd

# Model 3: scikit-learn LinearRegression.
sk_model = LinearRegression()
sk_model.fit(X_train, y_train)
coef_sk = sk_model.coef_.astype(float)
intercept_sk = float(sk_model.intercept_)
pred_sk = sk_model.predict(X_test)

def metric_row(name, intercept, coef, pred):
    return {
        "Method": name,
        "Intercept (b)": float(intercept),
        "CGPA coefficient (w₁)": float(coef[0]),
        "IQ coefficient (w₂)": float(coef[1]),
        "Test MSE": float(mean_squared_error(y_test, pred)),
        "Test R²": float(r2_score(y_test, pred)),
    }

results = pd.DataFrame([
    metric_row("Closed-form least squares", intercept_closed, coef_closed, pred_closed),
    metric_row("Gradient descent", intercept_gd, coef_gd, pred_gd),
    metric_row("Scikit-learn LinearRegression", intercept_sk, coef_sk, pred_sk),
])

st.subheader("1. Model performance on the test set")
m1, m2, m3 = st.columns(3)
for col, row in zip([m1, m2, m3], results.to_dict("records")):
    col.metric(row["Method"], f"R² = {row['Test R²']:.4f}")
    col.caption(f"MSE: {row['Test MSE']:.4f}")

st.dataframe(
    results.style.format({
        "Intercept (b)": "{:.6f}",
        "CGPA coefficient (w₁)": "{:.6f}",
        "IQ coefficient (w₂)": "{:.6f}",
        "Test MSE": "{:.6f}",
        "Test R²": "{:.6f}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.subheader("2. Visualise actual data and the fitted regression plane")
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    data["cgpa"], data["iq"], data["package"],
    alpha=0.55, label="Actual observations"
)
cgpa_range = np.linspace(float(data["cgpa"].min()), float(data["cgpa"].max()), 25)
iq_range = np.linspace(float(data["iq"].min()), float(data["iq"].max()), 25)
cgpa_grid, iq_grid = np.meshgrid(cgpa_range, iq_range)
package_grid = intercept_sk + coef_sk[0] * cgpa_grid + coef_sk[1] * iq_grid
ax.plot_surface(cgpa_grid, iq_grid, package_grid, alpha=0.35)
ax.set_xlabel("CGPA")
ax.set_ylabel("IQ")
ax.set_zlabel("Package (LPA)")
ax.set_title("Multivariate Linear Regression: CGPA, IQ and Package")
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

left, right = st.columns(2)
with left:
    st.markdown("**Gradient descent convergence**")
    fig_loss, ax_loss = plt.subplots(figsize=(7, 4))
    ax_loss.plot(np.arange(1, len(loss_history) + 1), loss_history)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Training MSE")
    ax_loss.set_title("Gradient Descent Loss")
    ax_loss.grid(True, alpha=0.25)
    fig_loss.tight_layout()
    st.pyplot(fig_loss)
    plt.close(fig_loss)

with right:
    st.markdown("**Actual vs predicted package**")
    fig_pred, ax_pred = plt.subplots(figsize=(7, 4))
    ax_pred.scatter(y_test, pred_sk, alpha=0.8)
    lower = float(min(y_test.min(), pred_sk.min()))
    upper = float(max(y_test.max(), pred_sk.max()))
    ax_pred.plot([lower, upper], [lower, upper], linestyle="--")
    ax_pred.set_xlabel("Actual package (LPA)")
    ax_pred.set_ylabel("Predicted package (LPA)")
    ax_pred.set_title("Actual vs Predicted — Scikit-learn")
    ax_pred.grid(True, alpha=0.25)
    fig_pred.tight_layout()
    st.pyplot(fig_pred)
    plt.close(fig_pred)

st.subheader("3. Predict a package for a new student")
col_a, col_b = st.columns(2)
with col_a:
    cgpa_value = st.number_input(
        "Student CGPA", min_value=0.0, max_value=10.0,
        value=float(np.clip(data["cgpa"].median(), 0.0, 10.0)), step=0.01
    )
with col_b:
    iq_value = st.number_input(
        "Student IQ", min_value=40, max_value=200,
        value=int(np.clip(round(data["iq"].median()), 40, 200)), step=1
    )

if st.button("Predict placement package", type="primary"):
    user_prediction = float(sk_model.predict(np.array([[cgpa_value, iq_value]], dtype=float))[0])
    st.metric("Predicted placement package", f"{user_prediction:.2f} LPA")
    if not (data["cgpa"].min() <= cgpa_value <= data["cgpa"].max()):
        st.warning("CGPA is outside the observed dataset range; this is extrapolation.")
    if not (data["iq"].min() <= iq_value <= data["iq"].max()):
        st.warning("IQ is outside the observed dataset range; this is extrapolation.")

st.subheader("4. Learned multivariate regression equation")
eq1, eq2, eq3 = st.columns(3)
eq1.markdown("**Closed-form**")
eq1.latex(
    f"y = {coef_closed[0]:.4f}x_1 "
    f"{'+' if coef_closed[1] >= 0 else '-'} {abs(coef_closed[1]):.4f}x_2 "
    f"{'+' if intercept_closed >= 0 else '-'} {abs(intercept_closed):.4f}"
)
eq2.markdown("**Gradient descent**")
eq2.latex(
    f"y = {coef_gd[0]:.4f}x_1 "
    f"{'+' if coef_gd[1] >= 0 else '-'} {abs(coef_gd[1]):.4f}x_2 "
    f"{'+' if intercept_gd >= 0 else '-'} {abs(intercept_gd):.4f}"
)
eq3.markdown("**Scikit-learn**")
eq3.latex(
    f"y = {coef_sk[0]:.4f}x_1 "
    f"{'+' if coef_sk[1] >= 0 else '-'} {abs(coef_sk[1]):.4f}x_2 "
    f"{'+' if intercept_sk >= 0 else '-'} {abs(intercept_sk):.4f}"
)

with st.expander("Gradient descent configuration and parameter comparison"):
    st.write(f"Training rows: {len(X_train)}")
    st.write(f"Testing rows: {len(X_test)}")
    st.write(f"Epochs: {int(epochs)}")
    st.write(f"Learning rate: {learning_rate}")
    st.write(f"Final gradient-descent training MSE: {loss_history[-1]:.6f}")
    st.write(
        "CGPA and IQ are standardised using training-set statistics for gradient descent. "
        "Displayed coefficients are converted back to the original CGPA and IQ scales."
    )

with st.expander("Observation and conclusion", expanded=True):
    st.markdown(
        f"""
**Observations**
- The model uses two independent variables: CGPA and IQ.
- The 3D scatter plot and regression plane visualise the relationship between the two features and package.
- Closed-form least squares and scikit-learn should produce nearly identical parameters.
- Gradient descent should approach comparable parameters when it converges.
- On the current split, scikit-learn test MSE is **{mean_squared_error(y_test, pred_sk):.4f}**
  and test R² is **{r2_score(y_test, pred_sk):.4f}**.

**Conclusion**

Multivariate linear regression was implemented using a closed-form least-squares solution,
gradient descent, and scikit-learn's `LinearRegression`. The model was evaluated using MSE
and R², and results were visualised with a 3D regression plane and an actual-versus-predicted
plot. The predicted package is an estimate based on the supplied dataset, not a guarantee.
"""
    )

st.caption(
    "Dataset: Cgpa_iq_package.csv • Features: CGPA and IQ • Target: package (LPA) • "
    "Interactive ML lab experiment"
)
