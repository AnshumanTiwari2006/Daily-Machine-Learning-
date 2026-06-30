# 🚀 Daily Machine Learning Journey

Welcome to my **Daily Machine Learning** repository! I am a B.Tech graduate actively deep-diving into the mathematical foundations, theories, and code implementations of Machine Learning algorithms. 

This repository serves as my personal pedagogical knowledge base. It contains Jupyter notebooks, from-scratch algorithmic implementations, and semi-detailed theory notes covering my daily learnings.

## 🎯 Repository Philosophy
My goal here is not just to call `model.fit()` and `model.predict()` using out-of-the-box libraries, but to genuinely understand **how** and **why** these algorithms work under the hood. 

Across these folders, you'll find:
- 🧮 **Mathematical Theory:** Breakdowns of core formulas, cost functions, and update rules.
- 🛠️ **From-Scratch Implementations:** Building algorithms from the ground up (often using raw NumPy) to prove the underlying mathematics.
- 🚀 **Scikit-Learn Comparisons:** Benchmarking my custom implementations against production-level libraries to validate accuracy and performance.

## 📂 Repository Structure

### 📉 1. Optimization Algorithms
The core engines that power modern machine learning and deep learning models.
- **[Gradients](./Gradients/):** The foundation of optimization. Covers Batch Gradient Descent, Stochastic Gradient Descent (SGD), and Mini-Batch Gradient Descent implementations from scratch.
- **[Gradient Descent for Regression](./Gradient_Descent_for_Regression/):** Applying the Gradient Descent mathematical update rules specifically to solve regression problems.

### 📈 2. Regression Techniques
Predicting continuous values through various linear and non-linear models.
- **[Linear Regression](./Linear_Regression/):** The starting point. Predicting a target variable using a single continuous feature.
- **[Multiple Regression](./Multiple_Regression/):** Extending core linear concepts to handle multiple independent variables.
- **[Polynomial Regression](./Polynomial_Regression/):** Modeling non-linear relationships by transforming features into higher-degree polynomial spaces.

### 🛡️ 3. Regularization Methods
Taming model complexity to prevent overfitting and improve generalization on unseen data.
- **[Ridge Regression](./Ridge_Regression/):** Applying L2 regularization to cleanly shrink coefficients and handle multicollinearity.
- **[Lasso Regression](./Lasso_Regression/):** Applying L1 regularization for simultaneous coefficient shrinkage and automated feature selection.
- **[ElasticNet Regression](./ElasticNet_Regression/):** The highly effective hybrid approach combining both L1 and L2 penalties.

### 🔮 4. Classification Models
Adapting regression principles to predict distinct, categorical outcomes.
- **[Logistic Regression](./Logistic_Regression/):** Explores the journey from primitive Perceptrons and Step Functions to the modern Sigmoid curve for binary classification probabilities.
- **[Softmax Regression](./Softmax_Regression/):** (Multinomial Logistic Regression) Generalizing Logistic Regression to elegantly classify data into multiple, mutually exclusive categories.

### 📏 5. Evaluation & Metrics
- **[Regression Metrics](./Regression_Metrics/):** Exploring the mathematical intuition and practical code behind measuring model success, including $R^2$ Score, MAE, MSE, and RMSE.
- **[Classification Metrics](./Metrics/):** Understanding how to evaluate classification models using Accuracy, the Confusion Matrix, and differentiating between Type I (False Positive) and Type II (False Negative) errors.
- **[ROC Curve & AUC](./ROC_Curve/):** Visualizing and quantifying a model's ability to distinguish between classes across all possible classification thresholds.



### 🧹 6. Data Preprocessing & Feature Engineering
Preparing raw data for robust machine learning models.
- **[Outlier Detection methods](./Outlier%20Detection%20methods/):** Statistical approaches to identify and handle anomalies, including Z-score (for normal distributions), IQR (for skewed data), and Percentile-based capping/trimming.
- **[Feature Scaling](./Feature%20Scaling/):** Leveling the mathematical playing field for gradient descent and distance-based algorithms using Standardization (Z-score) and Normalization (Min-Max scaling).


### 🗜️ 7. Dimensionality Reduction
Techniques for compressing high-dimensional data while retaining critical information.
- **[Principal Component Analysis (PCA)](./Principal%20Component%20Analysis/):** Reducing feature space to combat the curse of dimensionality, vastly speeding up algorithms like KNN on complex datasets (e.g., Image processing).

## 💡 How to Use This Repository
Each specific topic folder is fully modular and contains its own dedicated, deep-dive `README.md` alongside the Jupyter Notebooks. 

To explore a topic, navigate to its folder, read the theoretical README, and launch the `.ipynb` files to see the math translated directly into code!

---
*“You don't understand it until you can build it from scratch.”*
