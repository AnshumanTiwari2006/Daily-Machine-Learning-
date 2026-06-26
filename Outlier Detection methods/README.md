# 🚩 Outlier Detection and Removal Methods

Welcome to the **Outlier Detection** module! In real-world data, anomalies and extreme values (outliers) can severely skew your machine learning models, especially linear algorithms that are sensitive to variance. 

This directory focuses on the primary statistical techniques used to identify and handle outliers during the data preprocessing phase.

## 🎯 Overview

There is no single "best" way to detect an outlier; the right technique depends entirely on the underlying distribution of your data. This module breaks down three core methods:

1. **Z-score Method:** For Normally distributed (Gaussian) data.
2. **IQR Method:** For skewed data that does not follow a normal distribution.
3. **Percentile Method:** A non-parametric approach for capping extreme ends.

Across all notebooks, we explore both **Trimming** (completely dropping the outlier rows) and **Capping/Winsorization** (replacing outlier values with the boundary values).

## 📂 Contents & Findings

### 1. [`Outlier Detection and Removal using Z-score Method.ipynb`](Outlier%20Detection%20and%20Removal%20using%20Z-score%20Method.ipynb)
**Best for:** Data that follows a Normal (Gaussian) distribution.
* **Theory:** The Z-score tells you how many standard deviations away a data point is from the mean. 
* **Formula:** $z = \frac{x - \mu}{\sigma}$
* **Rule of Thumb:** Any point with a Z-score greater than $3$ or less than $-3$ (i.e., $|z| > 3$) is considered an outlier, as 99.7% of data should lie within 3 standard deviations in a normal distribution.
* **Implementation:** We calculate the mean and standard deviation, compute the bounds ($\mu \pm 3\sigma$), and then either trim or cap the outliers.

### 2. [`Outlier Detection and Removal using IQR Method.ipynb`](Outlier%20Detection%20and%20Removal%20using%20IQR%20Method.ipynb)
**Best for:** Skewed distributions (e.g., income, house prices) where the mean and standard deviation are heavily influenced by the outliers themselves.
* **Theory:** The Interquartile Range (IQR) measures the statistical dispersion of the middle 50% of the data.
* **Formula:** $IQR = Q_3 - Q_1$ (where $Q_1$ is the 25th percentile and $Q_3$ is the 75th percentile).
* **Rule of Thumb:** We establish lower and upper bounds using a $1.5 \times IQR$ multiplier:
  * Lower Bound: $Q_1 - 1.5 \times IQR$
  * Upper Bound: $Q_3 + 1.5 \times IQR$
* **Implementation:** We use boxplots to visually identify outliers, calculate the $Q_1$ and $Q_3$ values using NumPy/Pandas, and apply the bounds to filter the dataset.

### 3. [`Outlier Detection and Removal using Percentile Method.ipynb`](Outlier%20Detection%20and%20Removal%20using%20Percentile%20Method.ipynb)
**Best for:** Non-parametric bounds, or when you have domain knowledge dictating that a certain top/bottom percentage of data is invalid.
* **Theory:** Instead of using standard deviations or IQRs, we simply set arbitrary percentile thresholds to cut off the extreme tails of the data.
* **Rule of Thumb:** Commonly, the 1st percentile and 99th percentile are used. Anything below the 1st percentile or above the 99th percentile is treated as an outlier.
* **Implementation:** We utilize the `.quantile()` method in Pandas to find the exact values at our chosen percentiles, and then cap or trim the dataset accordingly.

## 🚀 Getting Started
Launch any of the notebooks to explore how these statistical bounds are computed and applied to clean a dataset:
```bash
jupyter notebook "Outlier Detection and Removal using IQR Method.ipynb"
```
