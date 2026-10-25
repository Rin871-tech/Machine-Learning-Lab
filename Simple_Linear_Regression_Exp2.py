# -*- coding: utf-8 -*-
"""
Experiment 2: Implementation of Simple Linear Regression
Machine Learning Lab Manual — Dr. Sachin Balawant Takmare

Interactive Streamlit UI using the professor-provided Placement dataset.
Expected CSV columns in the supplied dataset: cgpa, package

Install dependencies:
    python -m pip install streamlit numpy pandas matplotlib scikit-learn

Run:
    streamlit run Simple_Linear_Regression_Exp2.py

If "Placement (1).csv" is in the same folder, it loads automatically.
Otherwise, use the CSV uploader in the sidebar.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Page configuration and styling
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Experiment 2 | Simple Linear Regression",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
      .main-title {font-size: 2rem; font-weight: 750; margin-bottom: 0.2rem;}
      .subtitle {color: #64748b; margin-bottom: 1.2rem;}
      .metric-card {padding: 0.75rem 1rem; border: 1px solid #e2e8f0;
                    border-radius: 0.75rem; background: rgba(148,163,184,0.06);}
      div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22);
                                   padding: 0.8rem; border-radius: 0.7rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">📈 Experiment 2: Simple Linear Regression</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Compare the Normal Equation, Gradient Descent, and scikit-learn LinearRegression.</div>',
    unsafe_allow_html=True,
)

with st.expander("Aim, objectives and theory", expanded=False):
    st.markdown(
        """
**Aim:** To implement simple linear regression using Python and evaluate its performance.

**Objectives**
1. Study the concept of simple linear regression.
2. Compute model parameters using a closed-form solution and gradient descent.
3. Evaluate regression performance using MSE and R².
4. Visualise the regression results.

**Theory**

Simple linear regression models the relationship between one independent variable \(x\)
and one dependent variable \(y\):

\[
y = wx + b + \\varepsilon
\]

- \(w\): slope; \(b\): intercept; \(\\varepsilon\): error term.
- **Normal Equation:** obtains the least-squares parameters using a matrix solution.
  This app uses a pseudoinverse for numerical robustness.
- **Gradient Descent:** iteratively updates parameters to reduce mean squared error.
- **MSE:** average squared prediction error; lower is better.
- **R²:** goodness of fit; closer to 1 generally indicates a better fit.
"""
    )

with st.expander("Procedure followed", expanded=False):
    st.markdown(
        """
1. Import the required libraries and load the CSV dataset.
2. Inspect the dataset and visualise the independent and dependent variables.
3. Split the data into training and testing sets.
4. Train models using the Normal Equation, Gradient Descent, and scikit-learn.
5. Predict target values on the held-out test set.
6. Calculate MSE and R² for each model.
7. Plot the scatter data and fitted regression lines.
8. Enter a custom input to generate a prediction.
"""
    )


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------
def load_default_dataset():
    """Find the provided CSV next to this script, if present."""
    candidates = [
        Path(__file__).resolve().parent / "Placement (1).csv",
        Path(__file__).resolve().parent / "Placement.csv",
        Path.cwd() / "Placement (1).csv",
        Path.cwd() / "Placement.csv",
    ]
    for path in candidates:
        if path.exists():
            try:
                return pd.read_csv(path), str(path)
            except Exception:
                continue
    return None, None


def clean_numeric_columns(frame):
    """Return a copy with numeric-looking columns converted safely."""
    cleaned = frame.copy()
    for col in cleaned.columns:
        if cleaned[col].dtype == object:
            converted = pd.to_numeric(cleaned[col], errors="coerce")
            # Convert only if at least some non-missing values are numeric.
            if converted.notna().sum() > 0:
                cleaned[col] = converted
    return cleaned


default_df, default_source = load_default_dataset()

st.sidebar.header("⚙️ Dataset & model settings")
uploaded_file = st.sidebar.file_uploader("Upload your placement CSV", type=["csv"])

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        data_source = uploaded_file.name
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()
elif default_df is not None:
    raw_df = default_df
    data_source = default_source
else:
    st.info(
        "Upload the professor-provided **Placement (1).csv** file in the sidebar. "
        "The app expects numeric columns such as `cgpa` and `package`."
    )
    st.stop()

df = clean_numeric_columns(raw_df)
st.sidebar.caption(f"Loaded: {data_source}")

numeric_columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if len(numeric_columns) < 2:
    st.error("The CSV must contain at least two numeric columns: one input feature and one target.")
    st.stop()

# Use the exact column names from the professor's CSV when available.
default_x_index = numeric_columns.index("cgpa") if "cgpa" in numeric_columns else 0
default_y_index = numeric_columns.index("package") if "package" in numeric_columns else (1 if default_x_index == 0 else 0)

left, right = st.sidebar.columns(2)
with left:
    x_col = st.selectbox("Independent variable (X)", numeric_columns, index=default_x_index)
with right:
    y_col = st.selectbox("Dependent variable (y)", numeric_columns, index=default_y_index)

test_size = st.sidebar.slider("Test set (%)", min_value=10, max_value=40, value=20, step=5)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=9999, value=2, step=1)
learning_rate = st.sidebar.number_input(
    "Gradient descent learning rate", min_value=0.0001, max_value=1.0,
    value=0.05, step=0.005, format="%.4f",
)
iterations = st.sidebar.slider("Gradient descent iterations", 100, 20000, 5000, step=100)

if x_col == y_col:
    st.error("Choose different columns for X and y.")
    st.stop()

working = df[[x_col, y_col]].dropna().copy()
working = working[np.isfinite(working[x_col]) & np.isfinite(working[y_col])]
if len(working) < 5:
    st.error("At least five complete numeric rows are needed for this experiment.")
    st.stop()

X = working[[x_col]].to_numpy(dtype=float)
y = working[y_col].to_numpy(dtype=float)

if np.unique(X).size < 2:
    st.error("The independent variable must contain at least two distinct values.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100, random_state=int(random_state)
)

# ---------------------------------------------------------------------
# Models: Normal Equation, Gradient Descent, scikit-learn
# ---------------------------------------------------------------------
def fit_normal_equation(x_train, y_train):
    design = np.column_stack([x_train.ravel(), np.ones(len(x_train))])
    params = np.linalg.pinv(design) @ y_train
    return float(params[0]), float(params[1])


def fit_gradient_descent(x_train, y_train, rate, n_iterations):
    """
    Standardize X for stable optimization, then convert learned parameters
    back to the original X scale. y remains in its original scale.
    """
    x_values = x_train.ravel().astype(float)
    mean_x = float(np.mean(x_values))
    std_x = float(np.std(x_values))
    if std_x == 0:
        std_x = 1.0
    z = (x_values - mean_x) / std_x

    weight = 0.0
    bias = float(np.mean(y_train))
    n = len(y_train)
    losses = []

    # Batch gradient descent for MSE.
    for _ in range(int(n_iterations)):
        y_hat = weight * z + bias
        error = y_hat - y_train
        grad_w = (2.0 / n) * np.sum(error * z)
        grad_b = (2.0 / n) * np.sum(error)
        weight -= rate * grad_w
        bias -= rate * grad_b
        if not np.isfinite(weight) or not np.isfinite(bias):
            raise FloatingPointError(
                "Gradient descent became unstable. Try a smaller learning rate."
            )
        losses.append(float(np.mean((weight * z + bias - y_train) ** 2)))

    # Convert y = weight*((x - mean_x)/std_x) + bias into y = slope*x + intercept.
    slope = weight / std_x
    intercept = bias - slope * mean_x
    return float(slope), float(intercept), losses


try:
    ne_slope, ne_intercept = fit_normal_equation(X_train, y_train)
    gd_slope, gd_intercept, gd_losses = fit_gradient_descent(
        X_train, y_train, float(learning_rate), int(iterations)
    )
except FloatingPointError as exc:
    st.error(str(exc))
    st.stop()

sk_model = LinearRegression()
sk_model.fit(X_train, y_train)
sk_slope = float(sk_model.coef_[0])
sk_intercept = float(sk_model.intercept_)

models = {
    "Normal Equation": (ne_slope, ne_intercept),
    "Gradient Descent": (gd_slope, gd_intercept),
    "scikit-learn LinearRegression": (sk_slope, sk_intercept),
}

results = []
predictions_by_model = {}
for name, (slope, intercept) in models.items():
    predictions = slope * X_test.ravel() + intercept
    predictions_by_model[name] = predictions
    results.append({
        "Model": name,
        "Slope (w)": slope,
        "Intercept (b)": intercept,
        "MSE": mean_squared_error(y_test, predictions),
        "R²": r2_score(y_test, predictions) if len(y_test) > 1 else np.nan,
    })
results_df = pd.DataFrame(results)

# ---------------------------------------------------------------------
# UI: dataset, metrics, model results, visualisations
# ---------------------------------------------------------------------
st.subheader("1. Dataset overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows in CSV", f"{len(raw_df):,}")
m2.metric("Usable rows", f"{len(working):,}")
m3.metric("Training rows", f"{len(X_train):,}")
m4.metric("Testing rows", f"{len(X_test):,}")

with st.expander("View dataset", expanded=True):
    st.dataframe(working.head(20), use_container_width=True)
    st.caption(f"Selected relationship: {x_col} (X) → {y_col} (y)")
    st.write("Descriptive statistics")
    st.dataframe(working.describe().T, use_container_width=True)

st.subheader("2. Model evaluation on the test set")
st.caption("All three models are trained on the same training split and evaluated on the same held-out test rows.")
st.dataframe(
    results_df.style.format({
        "Slope (w)": "{:.6f}",
        "Intercept (b)": "{:.6f}",
        "MSE": "{:.6f}",
        "R²": "{:.6f}",
    }),
    use_container_width=True,
    hide_index=True,
)

best_row = results_df.loc[results_df["MSE"].idxmin()]
k1, k2, k3 = st.columns(3)
k1.metric("Lowest test MSE", best_row["Model"])
k2.metric("MSE", f"{best_row['MSE']:.4f}")
k3.metric("R²", f"{best_row['R²']:.4f}")

st.subheader("3. Visualise the data and regression lines")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X.ravel(), y, alpha=0.70, label="Dataset observations")
x_line = np.linspace(float(np.min(X)), float(np.max(X)), 250)
line_styles = ["-", "--", ":"]
for (name, (slope, intercept)), style in zip(models.items(), line_styles):
    ax.plot(x_line, slope * x_line + intercept, linestyle=style, linewidth=2, label=name)
ax.set_xlabel(x_col)
ax.set_ylabel(y_col)
ax.set_title(f"Simple Linear Regression: {x_col} vs {y_col}")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

with st.expander("Gradient Descent convergence", expanded=False):
    loss_fig, loss_ax = plt.subplots(figsize=(9, 4))
    loss_ax.plot(gd_losses)
    loss_ax.set_xlabel("Iteration")
    loss_ax.set_ylabel("Training MSE")
    loss_ax.set_title("Gradient Descent: MSE by Iteration")
    loss_ax.grid(True, alpha=0.25)
    loss_fig.tight_layout()
    st.pyplot(loss_fig)
    plt.close(loss_fig)
    st.write(f"Initial learning rate: {learning_rate}; iterations: {iterations}")
    st.write(f"Final recorded training MSE: {gd_losses[-1]:.6f}")

st.subheader("4. Predict using a custom input")
st.write(f"Enter a value for **{x_col}** to estimate **{y_col}**.")
min_x = float(np.min(X))
max_x = float(np.max(X))
median_x = float(np.median(X))
default_input = float(np.clip(median_x, min_x, max_x))
user_x = st.number_input(
    f"Enter {x_col}",
    min_value=min_x,
    max_value=max_x,
    value=default_input,
    format="%.4f",
    help="Input is limited to the observed dataset range to discourage unsupported extrapolation.",
)
selected_model = st.selectbox("Prediction model", list(models.keys()), index=2)
pred_slope, pred_intercept = models[selected_model]
user_prediction = pred_slope * user_x + pred_intercept

p1, p2, p3 = st.columns(3)
p1.metric("Input value", f"{user_x:.4f}")
p2.metric("Predicted value", f"{user_prediction:.4f}")
p3.metric("Selected model", selected_model)

st.info(
    "This is a simple linear model fitted to the supplied dataset. A predicted placement package "
    "is an estimate, not a guarantee; real placement outcomes depend on many variables not included here."
)

st.subheader("5. Observation and conclusion")
st.markdown(
    f"""
**Observations**
- The dataset contains **{len(working)} complete rows** for the selected X and y columns.
- The Normal Equation and scikit-learn estimate least-squares linear-regression parameters.
- Gradient Descent iteratively minimises training MSE; its result depends on learning rate and iteration count.
- On the selected test split, the lowest MSE is from **{best_row['Model']}** (MSE = {best_row['MSE']:.4f}, R² = {best_row['R²']:.4f}).

**Conclusion**

The experiment demonstrates simple linear regression with three approaches: a closed-form solution,
gradient descent, and scikit-learn's `LinearRegression`. Their slopes and intercepts can be compared,
test-set MSE and R² quantify predictive performance, and the plotted lines visualise the fitted
relationship between **{x_col}** and **{y_col}**.
"""
)

st.caption("Machine Learning Lab Manual by Dr. Sachin Balawant Takmare | Experiment 2")
