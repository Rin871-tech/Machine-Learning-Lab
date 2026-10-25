# -*- coding: utf-8 -*-
"""
Seaborn_42.py
Beginner-to-Advanced Guide to Statistical Data Visualization with Seaborn
Install: python -m pip install seaborn pandas numpy matplotlib
"""

# %% 1. INTRODUCTION
"""
Seaborn is a statistical visualization library built on top of Matplotlib.
It works naturally with pandas DataFrames and supports distribution plots,
categorical comparisons, relationships, regression visuals, and heatmaps.
"""

# %% 2. VERSION AND DATA
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Seaborn version:", sns.__version__)
tips = sns.load_dataset("tips")  # May need internet the first time.
print(tips.head())
print(tips.describe(include="all"))

# %% 3. HISTOGRAM AND KDE
sns.histplot(data=tips, x="total_bill", bins=25, kde=True)
plt.title("Restaurant Bill Distribution")
plt.show()

# %% 4. SCATTER PLOT WITH GROUPING
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time", style="smoker")
plt.title("Bill vs Tip")
plt.show()

# %% 5. LINE PLOT
flights = sns.load_dataset("flights")
sns.lineplot(data=flights, x="year", y="passengers", errorbar=None)
plt.title("Passengers Over Time")
plt.show()

# %% 6. BAR PLOT AND COUNT PLOT
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(data=tips, x="day", y="total_bill", errorbar=None, ax=axes[0])
axes[0].set_title("Average Bill by Day")
sns.countplot(data=tips, x="day", hue="time", ax=axes[1])
axes[1].set_title("Number of Records by Day and Time")
plt.tight_layout()
plt.show()

# %% 7. BOXPLOT AND VIOLIN PLOT
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(data=tips, x="day", y="total_bill", ax=axes[0])
sns.violinplot(data=tips, x="day", y="total_bill", inner="quartile", ax=axes[1])
axes[0].set_title("Bill Distribution: Box Plot")
axes[1].set_title("Bill Distribution: Violin Plot")
plt.tight_layout()
plt.show()

# %% 8. PAIRPLOT
iris = sns.load_dataset("iris")
sns.pairplot(iris, hue="species", corner=True)
plt.show()

# %% 9. CORRELATION HEATMAP
numeric = tips.select_dtypes(include="number")
correlation = numeric.corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Correlation Between Numeric Variables")
plt.show()

# %% 10. REGRESSION PLOT
sns.regplot(data=tips, x="total_bill", y="tip", scatter_kws={"alpha": 0.5})
plt.title("Tip vs Bill with Fitted Trend")
plt.show()

# %% 11. FACETGRID: MULTIPLE PANELS
grid = sns.FacetGrid(tips, col="time", row="smoker", margin_titles=True)
grid.map_dataframe(sns.scatterplot, x="total_bill", y="tip", alpha=0.7)
grid.set_axis_labels("Total bill", "Tip")
plt.show()

# %% 12. THEMES AND PALETTES
sns.set_theme(style="whitegrid", palette="deep", context="notebook")
sns.barplot(data=tips, x="day", y="tip", errorbar=None)
plt.title("Tips by Day")
plt.show()
sns.set_theme()  # Reset to defaults.

# %% 13. MISSING DATA VISUALIZATION
missing_demo = tips.copy()
missing_demo.loc[0:10, "tip"] = np.nan
sns.heatmap(missing_demo.isna(), cbar=False, yticklabels=False)
plt.title("Missing Values by Row and Column")
plt.show()

# %% 14. SAVE A FIGURE
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day", ax=ax)
fig.savefig("seaborn_scatter.png", dpi=300, bbox_inches="tight")
plt.show()

# %% 15. BEST PRACTICES
"""
- Use histplot/KDE for distributions, scatterplot for relationships,
  boxplot/violinplot for grouped distributions, and heatmap for matrices.
- Keep legends and axis labels readable.
- Correlation does not establish causation.
- Seaborn datasets may be downloaded on first use; for offline work, use local data.
- Seaborn returns Matplotlib objects, so you can further customize with Matplotlib.
"""

# %% 16. PRACTICE EXERCISES
"""
1. Plot total bill by day and split by smoker status.
2. Compare histograms of tips for lunch and dinner.
3. Build a correlation heatmap from a local CSV.
4. Use pairplot on a classification dataset.
5. Create a FacetGrid split by two categorical columns.
"""
print("Seaborn guide finished.")
