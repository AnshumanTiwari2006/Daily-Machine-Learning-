# Decision Trees

Welcome to the **Decision Tree** module! While algorithms like Logistic Regression rely on continuous mathematical equations and gradient descent, Decision Trees closely mimic human decision-making. They split data using a sequence of simple "If-Else" questions until a final conclusion is reached.

This directory dives deep into the architecture, mathematics, and code implementation of Decision Tree classifiers and regressors.

## Overview

A Decision Tree operates by recursively breaking down a dataset into smaller and smaller subsets based on feature conditions. The architectural structure consists of:
* **Root Node:** The very first, most mathematically important question that splits the entire dataset.
* **Decision Nodes:** Intermediate questions that further split the data based on new features.
* **Leaf Nodes:** The final output predictions (classes for Classification, or numerical averages for Regression).

**The Core Math:** How does the tree decide which question to ask first? It uses mathematical metrics to find the split that maximizes *Information Gain*:
1. **Entropy:** A measure of impurity, chaos, or randomness in the data. The tree actively tries to reduce Entropy to 0 (a perfectly pure leaf).
2. **Gini Impurity:** A computationally faster alternative to Entropy that measures the probability of misclassifying a random data point. Scikit-Learn uses Gini by default.

## Contents & Findings

### `Decision_Tree` & [`Decision_Tree_Part-2.ipynb`](Decision_Tree_Part-2.ipynb)
**Goal:** Build, visualize, and fundamentally understand the inner mechanics of Scikit-Learn's `DecisionTreeClassifier`.

* **The Process:** We train a Decision Tree on a dataset and explore how it non-linearly partitions the feature space into distinct geometric boxes.
* **Visualization:** We use `tree.plot_tree()` to explicitly draw the model graph. This allows us to visually inspect the exact Gini impurity, the exact mathematical threshold (e.g., `Age <= 25`), and the number of samples at every single node in the tree.
* **Overfitting (The Biggest Weakness):** By default, a Decision Tree will keep growing until every single leaf is 100% pure. We demonstrate how this leads to massive, highly complex trees that suffer from catastrophic overfitting on the training data.
* **Hyperparameter Tuning (Pruning):** To combat this extreme overfitting, we explore pre-pruning techniques by limiting the tree's architecture using hyperparameters like `max_depth` (limiting how deep the tree can grow) and `min_samples_split`.

## Getting Started
Launch the notebooks to learn how to train and visually interpret your own Decision Trees:
```bash
jupyter notebook "Decision_Tree_Part-2.ipynb"
```
