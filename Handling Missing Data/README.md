# Handling Missing Data

Welcome to the **Handling Missing Data** module! Real-world datasets are notoriously messy, and perhaps the single most common issue you will face as a Data Scientist is missing values (`NaN`s). Because almost no machine learning algorithms (like Linear Regression or Support Vector Machines) can inherently process mathematical operations on a `NaN` value, we must have robust strategies to deal with them before we can even begin training.

This directory provides a comprehensive deep-dive into how to analyze, clean, and mathematically impute missing data across both numerical and categorical features.

## Overview

There are broadly two paths when encountering missing data: **Delete it** or **Guess it (Impute it).** 

1. **Deletion (CCA):** Dropping rows or columns entirely. This is only statistically safe if the data is Missing Completely At Random (MCAR) and the missing percentage is very small (usually $< 5\%$).
2. **Imputation:** Mathematically estimating what the missing value *should* have been using simple statistical techniques (like Mean/Median), algorithmic modeling (KNN, Iterative), or randomized sampling.

## Contents & Findings

### 1. Deletion Strategies
* **[`Handling Missing Data Complete Case Analysis.ipynb`](Handling%20Missing%20Data%20Complete%20Case%20Analysis.ipynb)**
  * **Goal:** Understand when it is mathematically safe to simply drop rows containing NaNs.
  * **Key Concept:** We explore **MCAR** (Missing Completely At Random). If data is MCAR and affects less than 5% of the dataset, CCA (Complete Case Analysis) is acceptable. We visualize the data distribution (KDE plots) before and after deletion to strictly prove that the underlying shape of the data hasn't skewed.

### 2. Univariate Imputation (Statistical)
* **[`Handling Missing Data Using Imputation Part-1.ipynb`](Handling%20Missing%20Data%20Using%20Imputation%20Part-1.ipynb)** 
  * Imputing missing numerical values using baseline statistics via Scikit-Learn's `SimpleImputer`. (Mean is used for clean normal distributions; Median is used to resist heavy outliers).
* **[`Handling Missing Data of Categorical Data.ipynb`](Handling%20Missing%20Data%20of%20Categorical%20Data.ipynb)**
  * Handling missing text/categorical data using Mode (Most Frequent) imputation, or by replacing NaNs with a brand-new explicit category called `"Missing"`.
* **[`Handling Missing Data by Random Sample Imputation.ipynb`](Handling%20Missing%20Data%20by%20Random%20Sample%20Imputation.ipynb)**
  * Replacing NaNs with random samples drawn directly from the existing data to perfectly preserve the original statistical variance of the feature (something Mean imputation destroys).
* **[`Handling Missing Data by Missing Indicator.ipynb`](Handling%20Missing%20Data%20by%20Missing%20Indicator.ipynb)**
  * Creating a boolean feature (0 or 1) to explicitly signal to the model *that* a value was missing. Sometimes, the fact that data is missing carries heavy predictive weight (Missing Not At Random - MNAR).

### 3. Multivariate Imputation (Algorithmic)
* **[`Handling Missing Data via KNN Imputation.ipynb`](Handling%20Missing%20Data%20via%20KNN%20Imputation.ipynb)**
  * Using the K-Nearest Neighbors algorithm to find the $K$ rows that are most mathematically similar to the missing row, and averaging their values to seamlessly fill the NaN. Highly accurate but computationally expensive.
* **[`Handling Missing Data via Iterative Imputation.ipynb`](Handling%20Missing%20Data%20via%20Iterative%20Imputation.ipynb)**
  * (Also known as MICE - Multivariate Imputation by Chained Equations). Treats the missing feature column as a target variable ($y$) and uses all other features ($X$) to predict and fill the missing value via regression models. This is historically the most robust method for complex datasets.

## Getting Started
Launch the notebooks to explore how to rescue incomplete datasets:
```bash
jupyter notebook "Handling Missing Data Complete Case Analysis.ipynb"
```
