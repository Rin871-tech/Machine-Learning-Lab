# -*- coding: utf-8 -*-
"""
ScikitLearn_42.py
Complete Beginner-to-Advanced Learning Guide to scikit-learn
Inspired by the sectioned, example-driven style of Matplotlib_42.py.

HOW TO USE
1. Install: python -m pip install scikit-learn pandas numpy matplotlib
2. Run this file from top to bottom.
3. Each example uses small built-in or synthetic datasets.
4. Some examples are intentionally independent and repeat imports for clarity.
"""

# %% 1. INTRODUCTION
"""
scikit-learn is a Python machine-learning library for supervised learning,
unsupervised learning, preprocessing, model selection, and evaluation.

Common tasks:
- Classification: predict categories, such as spam/not spam.
- Regression: predict numeric values, such as house prices.
- Clustering: group similar observations without labels.
- Dimensionality reduction: represent data with fewer features.
- Preprocessing: scale values, encode categories, and handle missing data.
- Model selection: cross-validation and hyperparameter tuning.

Core vocabulary:
- X: feature matrix (rows = observations, columns = features).
- y: target/label.
- estimator: an object that learns from data.
- fit(): learn from training data.
- predict(): generate predictions.
- transform(): convert data using a learned transformation.
- pipeline: chain preprocessing and a model together.
"""

# %% 2. INSTALLATION AND VERSION
import sklearn
print("scikit-learn version:", sklearn.__version__)
print("Installation successful if the version printed above.")

# %% 3. FIRST CLASSIFICATION MODEL
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.25, random_state=42, stratify=iris.target
)
classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions, target_names=iris.target_names))
print("Confusion matrix:\n", confusion_matrix(y_test, predictions))

# %% 4. TRAIN/TEST SPLIT AND REPRODUCIBILITY
"""
Keep a test set separate from training. Use random_state to make a split
reproducible. For classification, stratify=y helps preserve class proportions.
Do not use the test set repeatedly to choose model settings.
"""

# %% 5. CLASSIFICATION: LOGISTIC REGRESSION WITH SCALING
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.25, random_state=7, stratify=iris.target
)
logistic_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=7)
)
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)
print("Logistic regression accuracy:", accuracy_score(y_test, y_pred))
print("Macro F1:", f1_score(y_test, y_pred, average="macro"))

# %% 6. REGRESSION: PREDICT A NUMERIC VALUE
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)
regressor = RandomForestRegressor(n_estimators=150, random_state=42)
regressor.fit(X_train, y_train)
reg_predictions = regressor.predict(X_test)
print("MAE:", mean_absolute_error(y_test, reg_predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, reg_predictions)))
print("R²:", r2_score(y_test, reg_predictions))

# %% 7. MODEL COMPARISON
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.25, random_state=42, stratify=iris.target
)
models = {
    "Dummy baseline": DummyClassifier(strategy="most_frequent"),
    "Decision tree": DecisionTreeClassifier(random_state=42),
    "Random forest": RandomForestClassifier(n_estimators=150, random_state=42),
    "SVM (scaled)": make_pipeline(StandardScaler(), SVC()),
    "Gradient boosting": GradientBoostingClassifier(random_state=42),
}
for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name:20s} accuracy={score:.3f}")

# %% 8. CROSS-VALIDATION
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    iris.data, iris.target, cv=5, scoring="accuracy"
)
print("CV scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())

# %% 9. HYPERPARAMETER TUNING
from sklearn.model_selection import GridSearchCV

parameter_grid = {
    "C": [0.1, 1, 10],
    "solver": ["lbfgs"],
}
search = GridSearchCV(
    LogisticRegression(max_iter=1000),
    parameter_grid, cv=5, scoring="accuracy"
)
search.fit(iris.data, iris.target)
print("Best parameters:", search.best_params_)
print("Best cross-validation score:", search.best_score_)

# %% 10. PREPROCESSING: MISSING VALUES AND SCALING
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

sample = pd.DataFrame({
    "age": [20, 25, np.nan, 40, 35],
    "income": [25000, 42000, 31000, np.nan, 55000],
})
imputer = SimpleImputer(strategy="median")
filled = pd.DataFrame(imputer.fit_transform(sample), columns=sample.columns)
scaled = pd.DataFrame(MinMaxScaler().fit_transform(filled), columns=sample.columns)
print("Original:\n", sample)
print("Imputed:\n", filled)
print("Scaled:\n", scaled)

# %% 11. CATEGORICAL DATA AND COLUMNTRANSFORMER
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

toy = pd.DataFrame({
    "age": [21, 35, 28, 45, 31, 52, 24, 41],
    "city": ["Pune", "Mumbai", "Pune", "Delhi", "Delhi", "Mumbai", "Pune", "Delhi"],
})
target = np.array([0, 1, 0, 1, 0, 1, 0, 1])
preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), ["age"]),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), ["city"]),
])
mixed_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000)),
])
mixed_pipeline.fit(toy, target)
print("Example prediction:", mixed_pipeline.predict(pd.DataFrame({"age": [30], "city": ["Pune"]})))

# %% 12. CLUSTERING WITH K-MEANS
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X_blobs, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.7, random_state=42)
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
cluster_ids = kmeans.fit_predict(X_blobs)
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=cluster_ids, s=25)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker="X", s=180, edgecolor="black")
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# %% 13. DIMENSIONALITY REDUCTION WITH PCA
from sklearn.decomposition import PCA

scaled_iris = StandardScaler().fit_transform(iris.data)
pca = PCA(n_components=2)
X_2d = pca.fit_transform(scaled_iris)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=iris.target)
plt.title("Iris Dataset Projected to 2 PCA Components")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.show()
print("Explained variance ratio:", pca.explained_variance_ratio_)

# %% 14. FEATURE IMPORTANCE
forest = RandomForestClassifier(n_estimators=200, random_state=42)
forest.fit(iris.data, iris.target)
importance = forest.feature_importances_
for name, value in sorted(zip(iris.feature_names, importance), key=lambda x: x[1], reverse=True):
    print(f"{name}: {value:.3f}")

# %% 15. ROC-AUC FOR BINARY CLASSIFICATION
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score, RocCurveDisplay

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.25, random_state=42, stratify=cancer.target
)
binary_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
binary_model.fit(X_train, y_train)
probabilities = binary_model.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, probabilities))
RocCurveDisplay.from_predictions(y_test, probabilities)
plt.title("ROC Curve")
plt.show()

# %% 16. SAVE AND LOAD A MODEL
import joblib

joblib.dump(logistic_model, "iris_logistic_pipeline.joblib")
loaded_model = joblib.load("iris_logistic_pipeline.joblib")
print("Loaded model prediction:", loaded_model.predict(iris.data[:3]))
print("Only load model files from trusted sources.")

# %% 17. COMMON METRICS CHEAT SHEET
"""
Classification:
- accuracy_score: overall fraction correct
- precision_score: how many predicted positives were correct
- recall_score: how many actual positives were found
- f1_score: harmonic mean of precision and recall
- roc_auc_score: ranking quality using predicted probabilities/scores

Regression:
- MAE: average absolute error
- MSE / RMSE: squared error / root squared error
- R²: proportion of target variance explained by the model

Clustering:
- silhouette_score: separation/cohesion estimate (higher is generally better)
"""

# %% 18. BEST PRACTICES AND COMMON MISTAKES
"""
BEST PRACTICES
- Split data before fitting transformations to avoid leakage.
- Put preprocessing and estimators in a Pipeline.
- Use cross-validation for model selection.
- Compare with a simple baseline.
- Choose metrics appropriate to the problem and class balance.
- Keep the final test set untouched until the end.
- Record random seeds and package versions.

COMMON MISTAKES
- Scaling or imputing before train/test split.
- Evaluating only on training data.
- Using accuracy alone on severely imbalanced data.
- Tuning repeatedly against the test set.
- Forgetting that labels and features must align row by row.
"""

# %% 19. PRACTICE EXERCISES
"""
1. Train KNN and compare it with logistic regression on Iris.
2. Try max_depth values 1, 2, 3, 5 for a decision tree.
3. Use cross-validation to compare three classifiers.
4. Build a pipeline for a CSV with numeric and categorical columns.
5. Train a regressor and report MAE, RMSE, and R².
6. Use K-Means on a dataset and explain why clusters may not match true labels.
7. Use GridSearchCV to tune a random forest.
8. Save a fitted pipeline and reload it for inference.
"""

print("Scikit-learn guide finished.")
