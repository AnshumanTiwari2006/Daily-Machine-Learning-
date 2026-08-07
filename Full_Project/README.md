# Titanic Survival Prediction End-to-End Project

This project provides an end-to-end workflow for analyzing and preparing the famous Titanic dataset for machine learning. The folder contains two distinct notebooks covering data visualization and data processing pipelines.

## Objective

The main goals of this project are:
- To perform Exploratory Data Analysis (EDA) and visualize survival rates based on different passenger characteristics.
- To handle missing values and preprocess features for modeling.
- To apply feature scaling and feature selection to prepare the data for predictive models.

## Project Structure

### 1. Exploratory Data Analysis
**File**: `Titanic_Dataset_End-to-End-Visualization.ipynb`
- Focuses on data exploration using `seaborn` and `matplotlib`.
- Visualizes distributions of numerical features (like Age and Fare).
- Explores the relationships between categorical variables (e.g., Sex, Class, Embarked) and passenger survival.
- Highlights missing values and basic dataset statistics.

### 2. Data Processing Pipeline
**File**: `Titanic_Dataset_End-to-End.ipynb`
- Demonstrates the data preprocessing pipeline using `pandas` and `scikit-learn`.
- **Missing Value Handling**: Imputes median values for `Age` and drops the `Cabin` column due to sparsity.
- **Feature Engineering**: Converts categorical labels like `Sex` into numerical formats and one-hot encodes the `Embarked` column. Drops irrelevant columns like `PassengerId`, `Name`, and `Ticket`.
- **Feature Scaling**: Applies `StandardScaler` to normalize numerical features.
- **Feature Selection**: Calculates and ranks the absolute correlation of all features with respect to the `Survived` target variable to identify the most predictive features.

## Libraries Used

- `pandas`: Data manipulation and missing value imputation.
- `numpy`: Numerical operations.
- `matplotlib` & `seaborn`: Data visualization.
- `scikit-learn`: Feature scaling (`StandardScaler`).

## How to Run

1. Clone the repository and navigate to the `Full_Project` directory.
2. Ensure you have the required dependencies installed:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Open either notebook using Jupyter to walk through the analysis and data preparation steps:
   ```bash
   jupyter notebook
   ```
