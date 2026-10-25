
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.inspection import permutation_importance


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Student Academic Performance Predictor")
st.caption(
    "Experiment No. 11 | Decision Tree vs Random Forest | Classification"
)

st.info(
    "Educational project. Predictions are estimates, not definitive "
    "judgments about an individual student's academic potential."
)


# =========================================================
# 2. LOAD AND CLEAN LOCAL DATASET
# =========================================================

DATA_PATH = Path(__file__).parent / "student-mat.csv"


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Place student-mat.csv in the same folder as app.py."
        )

    # The official UCI dataset uses semicolon delimiters.
    df = pd.read_csv(DATA_PATH, sep=";")

    # Normalize column names and remove duplicate records.
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates().copy()

    required_columns = {
        "G1", "G2", "G3", "school", "studytime",
        "absences", "age", "failures"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    # Remove rows with missing final grades.
    df = df.dropna(subset=["G3"]).copy()

    # Create classification target:
    # G3 >= 10 = Pass; G3 < 10 = Fail.
    df["Result"] = np.where(df["G3"] >= 10, "Pass", "Fail")

    # Prevent target leakage:
    # Do not use G1, G2, or G3 as input features.
    X = df.drop(columns=["G1", "G2", "G3", "Result"])
    y = df["Result"]

    return df, X, y


try:
    df, X, y = load_data()
except Exception as error:
    st.error("Unable to load the local student dataset.")
    st.exception(error)
    st.stop()


# =========================================================
# 3. TRAIN-TEST SPLIT AND MODEL TRAINING
# =========================================================

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

# Encoding is learned only from training data.
def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numerical",
                "passthrough",
                numerical_features,
            ),
        ],
        remainder="drop",
    )


# Both algorithms use precisely the same split.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


@st.cache_resource
def train_models(_X_train, _y_train):
    decision_tree = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=5,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    decision_tree.fit(_X_train, _y_train)
    random_forest.fit(_X_train, _y_train)

    return {
        "Decision Tree": decision_tree,
        "Random Forest": random_forest,
    }


models = train_models(X_train, y_train)


# =========================================================
# 4. EVALUATION METRICS
# =========================================================

def calculate_metrics(model):
    predictions = model.predict(X_test)

    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision (Pass)": precision_score(
            y_test, predictions, pos_label="Pass", zero_division=0
        ),
        "Recall (Pass)": recall_score(
            y_test, predictions, pos_label="Pass", zero_division=0
        ),
        "F1-score (Pass)": f1_score(
            y_test, predictions, pos_label="Pass", zero_division=0
        ),
    }


metrics = pd.DataFrame(
    {
        name: calculate_metrics(model)
        for name, model in models.items()
    }
).T


# =========================================================
# 5. SIDEBAR
# =========================================================

st.sidebar.title("Experiment 11")
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview & EDA",
        "Model Comparison",
        "Confusion Matrix",
        "Feature Importance",
        "Predict a Student",
    ],
)

st.sidebar.markdown("---")
st.sidebar.write("**Dataset:** UCI Student Performance")
st.sidebar.write("**Subject:** Mathematics")
st.sidebar.write(f"**Records:** {len(df)}")
st.sidebar.write("**Target:** Pass / Fail")
st.sidebar.write("**Test split:** 20%")
st.sidebar.write("**Random seed:** 42")


# =========================================================
# 6. OVERVIEW AND EDA
# =========================================================

if page == "Overview & EDA":

    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Student records", len(df))
    col2.metric("Input features", X.shape[1])
    col3.metric("Pass rate", f"{(y == 'Pass').mean():.1%}")
    col4.metric("Missing cells", int(df.isna().sum().sum()))

    st.subheader("Dataset preview")
    st.dataframe(df.head(15), use_container_width=True)

    st.subheader("Data quality")

    quality = pd.DataFrame(
        {
            "Data type": df.dtypes.astype(str),
            "Missing values": df.isna().sum(),
            "Unique values": df.nunique(),
        }
    )

    st.dataframe(quality, use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.subheader("Pass vs Fail distribution")

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(
            data=df,
            x="Result",
            order=["Fail", "Pass"],
            ax=ax,
        )
        ax.set_xlabel("Student outcome")
        ax.set_ylabel("Number of students")
        ax.set_title("Distribution of classification target")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Study time distribution")

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x="studytime", ax=ax)
        ax.set_xlabel("Weekly study time category")
        ax.set_ylabel("Number of students")
        ax.set_title("Students by study time")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Absences vs final grade")

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.scatterplot(
        data=df,
        x="absences",
        y="G3",
        hue="Result",
        alpha=0.7,
        ax=ax,
    )
    ax.set_xlabel("Number of absences")
    ax.set_ylabel("Final grade (G3)")
    ax.set_title("Absences and observed final grades")
    st.pyplot(fig)
    plt.close(fig)

    st.warning(
        "G1, G2 and G3 are excluded from the prediction inputs. "
        "G3 is used only to construct the target label."
    )


# =========================================================
# 7. MODEL COMPARISON
# =========================================================

elif page == "Model Comparison":

    st.header("Decision Tree vs Random Forest")

    st.write(
        "Both models are evaluated on the same held-out test set. "
        "Higher accuracy, precision, recall and F1-score are generally "
        "desirable, but the best model depends on the application."
    )

    st.subheader("Evaluation table")

    st.dataframe(
        metrics.style.format("{:.3f}"),
        use_container_width=True,
    )

    st.subheader("Metric comparison")

    fig, ax = plt.subplots(figsize=(10, 5))

    metrics.plot(
        kind="bar",
        ax=ax,
        ylim=(0, 1),
        rot=0,
    )

    ax.set_ylabel("Score")
    ax.set_xlabel("Algorithm")
    ax.set_title("Classification performance on the test set")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)

    st.pyplot(fig)
    plt.close(fig)

    best_f1 = metrics["F1-score (Pass)"].idxmax()
    best_accuracy = metrics["Accuracy"].idxmax()

    col1, col2 = st.columns(2)

    col1.metric("Higher accuracy", best_accuracy)
    col2.metric("Higher Pass F1-score", best_f1)

    st.markdown("### Interpretation")

    st.write(
        "**Decision Tree:** A single tree is easier to interpret, "
        "but it may overfit the training data."
    )

    st.write(
        "**Random Forest:** Combines predictions from many decision trees. "
        "Averaging across trees often improves generalization, although "
        "this is not guaranteed for every dataset."
    )


# =========================================================
# 8. CONFUSION MATRIX
# =========================================================

elif page == "Confusion Matrix":

    st.header("Confusion Matrix")

    chosen_model = st.selectbox(
        "Choose a model",
        list(models.keys()),
    )

    predictions = models[chosen_model].predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=["Fail", "Pass"],
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fail", "Pass"],
        yticklabels=["Fail", "Pass"],
        ax=ax,
    )

    ax.set_xlabel("Predicted outcome")
    ax.set_ylabel("Actual outcome")
    ax.set_title(f"{chosen_model} — Confusion Matrix")

    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        """
        **How to interpret it**

        - **True Fail:** The student actually failed and was predicted to fail.
        - **True Pass:** The student actually passed and was predicted to pass.
        - **False Pass:** The model predicted Pass, but the actual outcome was Fail.
        - **False Fail:** The model predicted Fail, but the actual outcome was Pass.
        """
    )

    with st.expander("Detailed classification report"):
        st.text(
            classification_report(
                y_test,
                predictions,
                labels=["Fail", "Pass"],
                zero_division=0,
            )
        )


# =========================================================
# 9. FEATURE IMPORTANCE
# =========================================================

elif page == "Feature Importance":

    st.header("Feature Importance")

    chosen_model = st.selectbox(
        "Choose a model",
        list(models.keys()),
    )

    model = models[chosen_model]

    st.write(
        "Permutation importance measures how much the model's test "
        "performance changes when one input feature is shuffled. "
        "It is calculated on the held-out test set."
    )

    with st.spinner("Calculating feature importance..."):
        importance_result = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="f1",
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
        )

    importance = pd.Series(
        importance_result.importances_mean,
        index=X_test.columns,
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 8))

    importance.plot(
        kind="barh",
        ax=ax,
    )

    ax.set_xlabel("Mean decrease in F1-score after shuffling")
    ax.set_ylabel("Input feature")
    ax.set_title(f"{chosen_model} — Permutation Importance")
    ax.axvline(0, linewidth=1)
    ax.grid(axis="x", alpha=0.25)

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Importance values can be negative or close to zero. "
        "They describe model behavior, not causation."
    )


# =========================================================
# 10. INTERACTIVE STUDENT PREDICTION
# =========================================================

elif page == "Predict a Student":

    st.header("Predict Student Outcome")

    st.write(
        "Enter student characteristics to compare predictions from "
        "both trained models. Do not enter grades G1, G2 or G3."
    )

    defaults = {
        "school": "GP",
        "sex": "F",
        "age": 17,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",
        "Medu": 3,
        "Fedu": 3,
        "Mjob": "other",
        "Fjob": "other",
        "reason": "course",
        "guardian": "mother",
        "traveltime": 1,
        "studytime": 2,
        "failures": 0,
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "yes",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
        "famrel": 4,
        "freetime": 3,
        "goout": 3,
        "Dalc": 1,
        "Walc": 2,
        "health": 4,
        "absences": 2,
    }

    user_input = {}

    with st.form("student_prediction_form"):

        st.subheader("Student and family information")

        columns = st.columns(3)

        for index, feature in enumerate(X.columns):

            with columns[index % 3]:

                if feature in categorical_features:

                    options = sorted(
                        X[feature].dropna().unique().tolist()
                    )

                    default_value = defaults.get(feature, options[0])

                    if default_value not in options:
                        default_value = options[0]

                    user_input[feature] = st.selectbox(
                        feature,
                        options,
                        index=options.index(default_value),
                    )

                else:

                    minimum = int(X[feature].min())
                    maximum = int(X[feature].max())

                    default_value = int(
                        defaults.get(
                            feature,
                            X[feature].median(),
                        )
                    )

                    default_value = min(
                        max(default_value, minimum),
                        maximum,
                    )

                    user_input[feature] = st.number_input(
                        feature,
                        min_value=minimum,
                        max_value=maximum,
                        value=default_value,
                        step=1,
                    )

        submitted = st.form_submit_button(
            "Predict student outcome",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        input_df = pd.DataFrame(
            [user_input],
            columns=X.columns,
        )

        st.subheader("Prediction results")

        result_columns = st.columns(2)

        for result_column, (name, model) in zip(
            result_columns,
            models.items(),
        ):

            predicted_class = model.predict(input_df)[0]

            probabilities = model.predict_proba(input_df)[0]

            pass_index = list(model.classes_).index("Pass")
            pass_probability = float(probabilities[pass_index])

            with result_column:

                st.markdown(f"### {name}")

                if predicted_class == "Pass":
                    st.success("Predicted outcome: PASS")
                else:
                    st.error("Predicted outcome: FAIL")

                st.metric(
                    "Model-estimated Pass probability",
                    f"{pass_probability:.1%}",
                )

                st.progress(pass_probability)

        st.caption(
            "These probabilities are model estimates, not guarantees "
            "of a student's future result. Predictions may be unreliable "
            "for students whose characteristics differ from this dataset."
        )


# =========================================================
# 11. FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Experiment No. 11 | Dataset: UCI Student Performance — Mathematics "
    "| Decision Tree Classifier vs Random Forest Classifier"
)
