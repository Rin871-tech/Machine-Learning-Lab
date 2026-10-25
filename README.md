Machine Learning Lab Experiments
A collection of Python-based Machine Learning laboratory experiments covering data visualization, regression, classification, preprocessing, and model evaluation. Selected experiments include interactive web applications built with Streamlit.
Experiments at a Glance
No.	Experiment	Main Concepts / Tools
1	Data Visualization and Data Analysis	Matplotlib, Pandas, Seaborn
2	Simple Linear Regression	Regression line, prediction, residuals, model evaluation
3	Multiple Linear Regression	Multiple input features, regression, prediction
4	Logistic Regression for Binary Classification	Binary classification, sigmoid function, classification metrics
5	Multiclass Classification	Multiclass Logistic Regression, One-vs-Rest, One-vs-One, classification metrics
6	Decision Tree Classification	Decision trees, Gini impurity, heart disease classification
7	Standardization	StandardScaler, feature scaling, placement data and wine data
8	Normalization	Min-Max scaling, feature ranges, preprocessing comparison
9	Linear Support Vector Machine (SVM)	Linear hyperplane, margins, banknote authentication
10	Non-Linear Support Vector Machine (SVM)	Polynomial and RBF kernels, non-linear boundaries, support vectors
11	Decision Tree and Random Forest	Classification and regression, student performance datasets, model comparison


Note: Experiment titles and numbering reflect the experiments prepared for this repository. If your official lab manual uses different numbering for standardization/normalization or the final Decision Tree/Random Forest practical, update the table to match it.

About the Repository
The repository contains practical implementations of common machine learning workflows:
- Loading and exploring datasets
- Data cleaning and preprocessing
- Visualizing data and relationships between variables
- Splitting datasets into training and testing sets
- Training regression and classification models
- Evaluating model performance using appropriate metrics
- Visualizing predictions, confusion matrices, and decision boundaries
- Testing models through interactive user interfaces where available
Technologies and Libraries
- Python 3
- Pandas — loading, cleaning, and manipulating tabular data
- NumPy — numerical operations
- Matplotlib — charts and plots
- Seaborn — statistical data visualization
- Scikit-learn — preprocessing, model training, predictions, and evaluation
- Streamlit — interactive web applications for selected experiments
- Git and GitHub — version control and project hosting
Experiment Details
Experiment 01 — Data Visualization and Data Analysis
Explore datasets and learn foundational visualization techniques.
- Pandas: load, inspect, filter, and summarize data.
- Matplotlib: create plots such as line charts, bar charts, histograms, and scatter plots.
- Seaborn: create statistical visualizations such as count plots, box plots, pair plots, and heatmaps.
- Interpret patterns, distributions, and relationships in datasets.
Experiment 02 — Simple Linear Regression
Implement a simple linear regression model to describe the relationship between one independent variable and one dependent variable.
- Fit a line to observed data.
- Make predictions for new input values.
- Visualize observed points and the fitted regression line.
- Evaluate predictions using appropriate regression metrics.
Experiment 03 — Multiple Linear Regression
Implement linear regression with multiple independent variables.
- Prepare features and a target variable.
- Split the dataset into training and testing sets.
- Train a multiple linear regression model.
- Generate predictions and evaluate model performance.
- Understand how multiple features contribute to the prediction.
Experiment 04 — Logistic Regression for Binary Classification
Implement logistic regression to classify observations into two classes.
- Train a binary classification model.
- Generate class predictions and, where available, class probabilities.
- Evaluate performance using accuracy, confusion matrix, precision, recall, and F1-score.
- Explore the classification workflow through the provided application/notebook.
Experiment 05 — Multiclass Classification
Explore classification tasks involving more than two classes.
- Work with multiclass datasets such as Iris or student-performance data, depending on the included version.
- Understand multiclass Logistic Regression and strategies such as One-vs-Rest (OvR) and One-vs-One (OvO).
- Explore classification approaches and compare their performance.
- Review metrics and visualizations, including decision boundaries where applicable.
Experiment 06 — Decision Tree Classification
Build a Decision Tree classifier using the heart disease dataset.
- Train a tree-based classification model.
- Explore Gini impurity and how a decision tree selects splits.
- Predict class labels for test samples.
- Evaluate the model using accuracy, a confusion matrix, and a classification report.
Experiment 07 — Standardization
Apply standardization to bring numerical features to a comparable scale.
- Use StandardScaler to transform features based on the training data.
- Compare feature distributions before and after scaling.
- Explore the supplied placement and wine datasets.
- Understand why scaling is important for many machine learning algorithms.
Experiment 08 — Normalization
Apply normalization to rescale numerical values, commonly to a fixed range such as 0 to 1.
- Use Min-Max scaling or the normalization technique implemented in the application.
- Compare original and transformed feature values.
- Observe how preprocessing affects feature ranges.
- Distinguish standardization from normalization.
Experiment 09 — Linear Support Vector Machine (SVM)
Implement a linear SVM for counterfeit banknote detection using the UCI Banknote Authentication dataset.
- Use numerical features extracted from banknote images.
- Scale input features and train an SVM with a linear kernel.
- Evaluate the classifier with accuracy, a confusion matrix, and a classification report.
- Explore feature distributions and try a sample prediction.
The current application is implemented with Streamlit. The problem statement mentions Flask and HTML; use a Flask-based interface if that framework is specifically required for submission.
Experiment 10 — Non-Linear Support Vector Machine (SVM)
Compare non-linear SVM kernels on datasets that cannot be separated effectively by a straight line.
- Generate non-linear data using make_moons or make_circles.
- Train SVM classifiers with Polynomial and Radial Basis Function (RBF) kernels.
- Adjust parameters such as C, gamma, and polynomial degree.
- Compare accuracy, confusion matrices, classification reports, decision boundaries, and support vectors.
Experiment 11 — Decision Tree and Random Forest
Explore tree-based models for classification and regression.
- Build classification and regression workflows using the datasets included with the experiment, including student-performance datasets where applicable.
- Train Decision Tree and Random Forest models.
- Compare predictions and model performance using suitable classification or regression metrics.
- Visualize and interpret model results through the provided interface.
Setup
Requirements
Install Python 3 and pip. Some applications may need an internet connection to download a dataset on their first run.
Create and activate a virtual environment (recommended)
macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
Windows
python -m venv .venv
.venv\\Scripts\\activate
Install dependencies
Install the requirements file supplied with the experiment you want to run. For example:
python -m pip install -r requirements_svm.txt
Each experiment may have a separate requirements file, such as requirements_exp2.txt, requirements_multiple_regression.txt, requirements_logistic_regression.txt, requirements_multiclass.txt, requirements_decision_tree.txt, requirements_scaling.txt, or requirements_knn.txt.
Running the Applications
Run the Python application corresponding to the experiment. For Streamlit apps, use:
python -m streamlit run Linear_SVM_Banknote_Exp09.py
For Experiment 10:
python -m streamlit run Nonlinear_SVM_Exp10.py
For other experiments, replace the filename with the relevant .py file in the repository. Streamlit prints a local URL in the terminal; open it in your browser.
Common Evaluation Metrics
The metric depends on whether the experiment is a regression or classification task.

Classification
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report
Regression
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)
Only report metrics that are actually calculated by the corresponding experiment.
Suggested Repository Structure
The following is an example. Keep the names and folders aligned with the actual files in your repository.
Machine-Learning-Lab/
├── README.md
├── requirements_*.txt
├── *.py
├── *.ipynb
└── *.csv
Learning Outcomes
By completing these experiments, students gain practical experience with:
- Data exploration and visualization
- Simple and multiple linear regression
- Binary and multiclass classification
- Decision Tree, Random Forest, KNN, and SVM algorithms
- Feature standardization and normalization
- Training/testing workflows and model evaluation
- Interpreting plots, predictions, and classification boundaries
- Building interactive machine learning applications
Disclaimer
These experiments are intended for academic and educational use. Predictions from example applications are not guarantees for real-world medical, financial, or authentication decisions.
Author
 Vaishnavi Chavan
Machine Learning Laboratory Experiments
