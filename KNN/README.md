# 📍 K-Nearest Neighbors (KNN)

Welcome to the **K-Nearest Neighbors (KNN)** module! KNN is one of the simplest, most intuitive, and historically significant machine learning algorithms. It is a non-parametric, lazy learning algorithm heavily utilized for classification (and occasionally regression).

This directory focuses on the theoretical intuition and practical code implementation of KNN.

## 🎯 Overview

KNN operates on a very simple, real-world premise: *"Show me your neighbors, and I'll tell you who you are."*

* **The Logic:** To predict the class of a brand-new data point, the algorithm calculates the mathematical geometric distance (usually Euclidean distance) between that new point and every single other point in the training dataset.
* **The 'K':** It then isolates the $K$ closest points (the "nearest neighbors") and takes a majority democratic vote to determine the final classification of the new point.
* **The Scaling Requirement:** Because KNN fundamentally relies on raw distance metrics, it is **absolutely critical** to scale your features (e.g., using Standardization) before training the model. If you skip this, features with massive numerical ranges will unjustly dominate and ruin the distance calculations.

## 📂 Contents & Findings

### [`KNN_Part-1.ipynb`](KNN_Part-1.ipynb)
**Goal:** Implement, preprocess, and train a KNN classifier on real-world medical data.

* **The Dataset:** We utilize the well-known `breast-cancer.csv` dataset. The objective is to accurately predict whether a tumor diagnosis is Malignant (M) or Benign (B) based on 30 different continuous numeric features (such as tumor radius, texture, area, and smoothness).
* **Preprocessing Pipeline:**
  * We cleanly separate the independent numeric features ($X$) from the target diagnosis label ($y$).
  * **Crucial Step:** We strictly apply Scikit-Learn's `StandardScaler` to standardize both the training and testing data. Because tumor `area_mean` is measured in the thousands while `smoothness_mean` is a tiny decimal, failing to scale would cause the model to incorrectly assume that `area` is the only feature that matters.
* **Model Implementation:** We construct and train Scikit-Learn's highly optimized `KNeighborsClassifier`, explicitly setting $K=3$ (`n_neighbors=3`) to classify the testing data.

## 🚀 Getting Started
Launch the notebook to explore the fundamentals of distance-based classification:
```bash
jupyter notebook "KNN_Part-1.ipynb"
```
