# Hyperparameter Tuning: GridSearchCV

Welcome to the **GridSearchCV** module! After choosing a machine learning algorithm (like KNN or Logistic Regression) and feeding it clean, mathematically robust data, there is still one massive step left before deployment: **Hyperparameter Tuning**.

This directory focuses on computationally finding the absolute *perfect* mathematical settings for your model using Scikit-Learn's powerful `GridSearchCV` class.

## Overview

In machine learning, there is a very strict difference between **Parameters** and **Hyperparameters**:
* **Parameters:** Values the mathematical model learns completely on its own during the training phase (like the weights/coefficients $W$ and bias $b$ in Linear Regression).
* **Hyperparameters:** Settings *you* (the Data Scientist) must manually define before training even starts (like the $K$ in K-Nearest Neighbors, or the $C$ penalty strength in Logistic Regression).

**GridSearchCV** completely automates the tedious process of finding the best hyperparameters. It exhaustively trains the model on every single possible combination of the hyperparameters you provide, evaluates them using robust **Cross-Validation**, and mathematically proves which combination is superior.

## Contents & Findings

### [`GridSearchCV.ipynb`](GridSearchCV.ipynb)
**Goal:** Prove how algorithmically optimizing hyperparameters can dramatically boost a model's baseline accuracy.

* **The Problem:** When we instantiated `KNeighborsClassifier(n_neighbors=5)`, why did we blindly choose 5? Was 7 better? Was 3 better? Manually running a custom Python `for` loop to test this is messy, prone to data leakage, and incredibly inefficient.
* **The Solution (GridSearchCV):** 
  * We construct a parameter dictionary (the "Grid")—for example, testing $K$ values ranging from 1 to 30, and simultaneously testing different mathematical distance metrics (like `euclidean` vs `manhattan`).
  * We pass the raw model and the parameter grid directly into `GridSearchCV(cv=5)`.
  * **Cross-Validation (`cv=5`):** The algorithm automatically slices the training data into 5 distinct chunks, constantly rotating which chunk acts as the validation set. This mathematically guarantees that our chosen hyperparameters don't accidentally overfit to a specific, "lucky" train/test split.
* **The Result:** After the exhaustive search, we use `grid.best_params_` to instantly uncover the optimal settings, and `grid.best_score_` to verify the cross-validated accuracy boost.

## Getting Started
Launch the notebook to learn how to aggressively squeeze every last drop of accuracy out of your machine learning models:
```bash
jupyter notebook "GridSearchCV.ipynb"
```
