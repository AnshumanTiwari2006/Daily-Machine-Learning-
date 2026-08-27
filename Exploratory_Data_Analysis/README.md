# Exploratory Data Analysis (EDA)

Welcome to the **Exploratory Data Analysis (EDA)** module! Before engineering features or training any machine learning models, you must act as a statistical detective. EDA is the absolutely foundational step where we mathematically and visually investigate a dataset to uncover underlying patterns, spot anomalies, and formulate hypotheses.

While tools like Pandas Profiling can automate parts of this process, performing manual EDA is a critical, irreplaceable skill for answering highly specific, nuanced business questions about your data.

## Overview

EDA is fundamentally divided into two core analytical approaches:

1. **Univariate Analysis:** Analyzing a single feature (variable) entirely independently to deeply understand its distribution, central tendency, and spread (variance).
2. **Multivariate Analysis:** Analyzing the mathematical relationships, interactions, and correlations between two or more features simultaneously.

## Contents & Findings

### 1. [`EDA using Univariate Analysis.ipynb`](EDA%20using%20Univariate%20Analysis.ipynb)
**Goal:** Understand the independent behavior of single variables within the dataset.
* **Categorical Data:** We use Count Plots (`sns.countplot()`) and Pie Charts to visualize the frequency and percentage makeup of distinct text categories (e.g., *How many passengers survived vs. died?*).
* **Numerical Data:** We use Histograms (`sns.histplot()`) to explicitly view frequency distributions, KDE plots (`sns.kdeplot()`) to approximate smooth probability density curves, and Box Plots (`sns.boxplot()`) to instantly identify mathematical outliers based on the Interquartile Range (IQR).

### 2. [`EDA using Multivariate Analysis.ipynb`](EDA%20using%20Multivariate%20Analysis.ipynb)
**Goal:** Discover hidden correlations and complex relationships by comparing multiple variables against each other.
* **Numerical vs. Numerical:** We use Scatter Plots (`sns.scatterplot()`) to find linear or non-linear relationships between two continuous variables (e.g., *Age vs. Fare*).
* **Numerical vs. Categorical:** We use Bar Plots (`sns.barplot()`) to find the average of a number grouped by a category, and grouped Box Plots to compare outlier distributions across different classes.
* **Complex Interactions:** We utilize Heatmaps (`sns.heatmap()`) paired with the Pandas `.corr()` function to visualize the mathematical correlation matrix of the entire dataset at a glance. We also leverage Pair Plots (`sns.pairplot()`) to generate a massive grid of scatter plots, instantly mapping the relationship of every feature against every other feature.

## Getting Started
Launch the notebooks to dive into the art of manual data investigation using Pandas, Matplotlib, and Seaborn:
```bash
jupyter notebook "EDA using Univariate Analysis.ipynb"
```
