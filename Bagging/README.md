# 🛍️ Ensemble Learning: Bagging (Bootstrap Aggregating)

Welcome to the **Bagging** module! While a basic Voting Ensemble combines completely different algorithms (like KNN + SVM + Logistic Regression), Bagging takes a different approach: it combines hundreds of the *exact same* algorithm (usually Decision Trees) to create a highly accurate, robust model.

This directory dives into the mechanics of **Bootstrap Aggregating (Bagging)**, which forms the mathematical foundation of incredibly powerful algorithms like the Random Forest.

## 🎯 Overview

Decision Trees are highly interpretable and mathematically powerful, but they suffer from one massive flaw: **High Variance (Overfitting)**. If you change just a few rows of training data, the entire structure of the tree changes wildly. 

**Bagging solves this through two mathematical steps:**
1. **Bootstrapping (Row Sampling):** Instead of giving a model the entire dataset, we randomly sample rows *with replacement* to create hundreds of slightly different, unique mini-datasets.
2. **Aggregating:** We train an independent Decision Tree on every single mini-dataset in parallel. To make a final prediction, we aggregate (majority vote for classification, or mean average for regression) the answers from all hundreds of trees.

Because each tree is trained on slightly different data, they make completely different mathematical errors. When we average them all together, those individual errors cancel out, drastically reducing the model's variance without sacrificing accuracy.

## 📂 Contents & Findings

### 1. [`Bagging_Part-1.ipynb`](Bagging_Part-1.ipynb)
**Goal:** Build and analyze a Bagging Classifier using Scikit-Learn to prove how it fundamentally dominates standalone models.

* **The Process:** We explicitly train a standalone `DecisionTreeClassifier` and compare its decision boundaries against a `BaggingClassifier` (which uses hundreds of Decision Trees under the hood).
* **Key Finding:** The standalone Decision Tree heavily overfits the training data, drawing jagged boundaries to capture mathematical noise. The Bagging Classifier, however, achieves a much smoother, robust decision boundary that dramatically increases accuracy on the unseen test set.
* **Out-of-Bag (OOB) Score:** Because bootstrapping samples *with replacement*, mathematically, roughly 37% of the original data is *never seen* by any given tree. We demonstrate how to use this left-out data as a free, built-in validation set to measure model accuracy without strictly needing a train-test split (`oob_score=True`).

### 2. [`Bagging_Part-2.ipynb`](Bagging_Part-2.ipynb)
**Goal:** Explore the impact of using different base estimators (like Support Vector Machines) inside a Bagging architecture.

* **The Process:** While Bagging is famous for using Decision Trees (which leads to Random Forests), the `BaggingClassifier` can actually accept *any* algorithm. We instantiate a Bagging Classifier using `SVC` (Support Vector Classifier) as the base estimator.
* **Key Finding:** We analyze the computational cost and accuracy differences when bagging non-tree algorithms. We learn that while Bagging mathematically reduces the variance of *high-variance* models (like Trees), applying it to highly stable models (like SVMs) often yields diminishing returns in accuracy while massively increasing training time.

### 3. [`Bagging_Part-3.ipynb`](Bagging_Part-3.ipynb)
**Goal:** Apply Bootstrap Aggregating to continuous numerical predictions using the `BaggingRegressor`.

* **The Process:** We pivot from Classification to Regression by tackling the California Housing dataset. We train a `BaggingRegressor` using `DecisionTreeRegressor` as the base estimator.
* **Key Finding:** Just as Bagging smooths out jagged classification boundaries, it mathematically smooths out the extreme, noisy predictions of standalone regression trees. By averaging hundreds of predicted numerical values, the ensemble drastically reduces the overall Mean Squared Error (MSE) and severely limits the impact of outliers.

## 🚀 Getting Started
Launch the notebook to mathematically witness how Bagging dramatically stabilizes complex algorithms:
```bash
jupyter notebook "Bagging_Part-1.ipynb"
```
