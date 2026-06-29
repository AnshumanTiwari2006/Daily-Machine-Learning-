# 📉 Principal Component Analysis (PCA)

Welcome to the **Principal Component Analysis (PCA)** module! Working with high-dimensional data (like images) can drastically slow down machine learning algorithms and lead to the dreaded "curse of dimensionality."

This directory explores PCA, an incredibly powerful unsupervised mathematical technique used to reduce the number of features in a dataset while retaining as much of the critical variance (information) as possible.

## 🎯 Overview

PCA works by identifying the "principal components" of the data—new, uncorrelated variables that successively maximize variance. 
* **Dimensionality Reduction:** It compresses data from hundreds or thousands of features into a much smaller, dense feature space.
* **Feature Extraction:** It doesn't just arbitrarily drop columns; it mathematically transforms and projects existing columns into new, optimized components.
* **Performance Boost:** By heavily reducing the dimensions, distance-based algorithms like K-Nearest Neighbors (KNN) can run significantly faster without sacrificing much accuracy.

## 📂 Contents & Findings

### [`PCA on Digit Recognizer.ipynb`](PCA%20on%20Digit%20Recognizer.ipynb)
**Goal:** Prove the power of PCA by compressing 784 raw pixel features of handwritten digits and observing the impact on a KNN classifier.

* **The Dataset:** We use the `Digit_Recognizer_train.csv` dataset (a classic MNIST-style problem). Each image is $28 \times 28$ pixels, meaning every single image has $784$ unique features.
* **Baseline Model:** 
  * We train a standard `KNeighborsClassifier` on the raw, uncompressed 784-dimensional data. 
  * We achieve a strong baseline accuracy (~96.4%), but note the significant time it takes KNN to compute distances in such a high-dimensional space.
* **Applying PCA:**
  * We strictly apply `StandardScaler` first, as PCA relies on variance and is highly sensitive to differing feature scales.
  * We use Scikit-Learn's `PCA(n_components=200)` to drastically compress the feature space from 784 dimensions down to just 200.
* **The Results:**
  * We train a new KNN model on the mathematically compressed `X_train_new` data.
  * The accuracy remains remarkably high (~95.0%), but the computational speed is vastly improved.
  * We also loop through various `n_components` (from 1 to 20) to print the accuracy at each step, clearly showing how quickly the model regains its predictive power as critical principal components are added.

## 🚀 Getting Started
Launch the notebook to explore how PCA handles and compresses high-dimensional image data:
```bash
jupyter notebook "PCA on Digit Recognizer.ipynb"
```
