# 🛠️ Machine Learning Pipelines

Welcome to the **Pipeline** module! In real-world machine learning, training a model involves a strict sequence of interdependent steps: imputing missing values, encoding categorical text, scaling numeric features, selecting the best features, and finally passing the cleaned data to the model. 

Running these steps manually (one by one) for both the training and testing sets is highly prone to errors, especially **data leakage**. This directory focuses on using Scikit-Learn's `Pipeline` class to architect clean, leak-proof, and production-ready machine learning workflows, and importantly, how to export and deploy them.

## 🎯 Overview

A Pipeline sequentially applies a list of data transformers and caps it off with a final estimator (the model). 

* **Preventing Data Leakage:** When you manually use `fit_transform()` on training data and `transform()` on testing data, it's very easy to accidentally fit on the test data or leak mathematical information. A Pipeline handles this logic internally, ensuring absolutely zero data leakage.
* **Production Ready (Pickling):** Instead of saving an imputer, an encoder, a scaler, and a model separately into multiple `pickle` files for deployment, you can wrap them all inside a single Pipeline object. When deployed, you just pass raw user data into that single object.
* **Clean Code:** Pipelines drastically reduce the number of lines of code needed for complex preprocessing, making the codebase highly readable and modular.

## 📂 Contents & Findings

### 1. [`ML Pipeline.ipynb`](ML%20Pipeline.ipynb)
**Goal:** Build, train, and export an end-to-end Machine Learning pipeline that handles raw data natively.
* **The Architecture:** We design a complex multi-step workflow. We chain multiple preprocessing blocks (`SimpleImputer`, `OneHotEncoder`, `MinMaxScaler`, and `SelectKBest` via `ColumnTransformer`) directly into a final predictive estimator (`DecisionTreeClassifier`).
* **Implementation:** 
  * We use Scikit-Learn's `Pipeline` class to rigidly define and name each step of the journey.
  * We explore the `.fit()` and `.predict()` methods of the Pipeline, demonstrating how it seamlessly routes data through all transformations before arriving at the model.
  * **Crucial Step:** We use the Python `pickle` library to export the completely fitted pipeline into a single file (`pipe.pkl`), perfectly encapsulating the entire preprocessing and modeling logic.

### 2. [`Pipeline Prediction.ipynb`](Pipeline%20Prediction.ipynb)
**Goal:** Simulate a production (deployment) environment by loading the exported pipeline and making predictions on completely unseen, raw user input.
* **The Process:** We load the `pipe.pkl` file back into memory using `pickle.load()`.
* **The Power of Pipelines:** We simulate a user submitting raw data (as an array). Because our pipeline inherently remembers exactly *how* to impute, scale, and encode from the original training phase, we simply pass the raw data straight into `pipe.predict()`. The pipeline automatically cleans the raw text/numbers and outputs the final classification, proving how effortless deployment becomes.

## 🚀 Getting Started
Launch the training notebook to learn how to architect robust ML pipelines:
```bash
jupyter notebook "ML Pipeline.ipynb"
```
