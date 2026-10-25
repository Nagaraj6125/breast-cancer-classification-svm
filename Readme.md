# Breast Cancer Classification Using Support Vector Machines (SVM)

## Overview

This project implements a machine learning classification system to distinguish between **benign and malignant breast tumors** using the Breast Cancer Wisconsin (Diagnostic) dataset.

The project explores Support Vector Machines (SVM), compares Linear and RBF kernels, and applies feature scaling and hyperparameter tuning to improve classification performance.

The objective is to understand how SVM works on a real-world medical dataset and evaluate the model using appropriate classification metrics.

> **Disclaimer:** This project is for educational and machine learning experimentation purposes only. It is not intended for clinical diagnosis or medical decision-making.

## Objectives

- Build a binary classification model for benign and malignant tumors.
- Understand Linear and RBF SVM kernels.
- Apply feature scaling using `StandardScaler`.
- Compare model performance using classification metrics.
- Tune hyperparameters using cross-validation.
- Evaluate malignant tumor detection using precision, recall, and F1-score.

## Dataset

**Breast Cancer Wisconsin (Diagnostic) Dataset**

- **Source:** [Kaggle – Breast Cancer Wisconsin (Diagnostic) Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Samples:** 569
- **Original columns:** 32, including the identifier and diagnosis columns
- **Predictive features:** 30 numerical features
- **Target variable:** `diagnosis`
- **Classes:** Benign (B) and Malignant (M)

The features describe characteristics of cell nuclei extracted from breast mass images, including radius, texture, perimeter, area, smoothness, and other measurements.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Workflow

### 1. Data Loading and Exploration

- Loaded the dataset using Pandas.
- Inspected the dataset shape, column names, data types, and target distribution.
- Identified unnecessary columns.

### 2. Data Preprocessing

- Removed the empty `Unnamed: 32` column.
- Removed the `id` column because it is an identifier rather than a predictive measurement.
- Encoded the target variable:
  - `B` → `0` (Benign)
  - `M` → `1` (Malignant)
- Separated the features and target variable.
- Split the data into training and testing sets.

### 3. Feature Scaling

Applied `StandardScaler` to standardize numerical features.

The scaler was fitted on the training data and then used to transform both the training and testing data, preventing information leakage from the test set.

### 4. Model Development

The following SVM approaches were explored:

- **Linear SVM:** Uses a linear decision boundary.
- **RBF SVM:** Uses a Radial Basis Function kernel to model nonlinear decision boundaries.
- **Hyperparameter tuning:** Evaluated different settings, including the regularization parameter `C`, using 5-fold cross-validation.

### 5. Model Evaluation

The models were evaluated using:

- Accuracy
- Confusion matrix
- Precision
- Recall
- F1-score
- Cross-validation recall

Particular attention was given to **malignant-class recall**, since missed malignant cases are an important error to understand in this dataset.

## Results

### Model Performance Comparison

| Model | Test Accuracy | Malignant Recall |
|---|---:|---:|
| Linear SVM | 96.49% | 90.48% |
| RBF SVM | 97.37% | 92.86% |
| Tuned RBF SVM | 97.37% | 92.86% |

The tuned RBF SVM used `C = 10` and `gamma = 'scale'`. Its 5-fold cross-validation recall was approximately **95.88%**.

### Tuned RBF SVM: Confusion Matrix

Rows represent actual classes and columns represent predicted classes. The class order is Benign, followed by Malignant.

```text
[[72  0]
 [ 3 39]]
```

Interpretation:

- **72:** Benign cases correctly classified as benign.
- **0:** Benign cases incorrectly classified as malignant.
- **3:** Malignant cases incorrectly classified as benign.
- **39:** Malignant cases correctly classified as malignant.

### Tuned RBF SVM: Classification Metrics

| Metric | Benign | Malignant |
|---|---:|---:|
| Precision | 96% | 100% |
| Recall | 100% | 92.86% |
| F1-score | 98% | 96% |
| Support | 72 | 42 |

The tuned model achieved **97.37% test accuracy**, with 100% precision and 92.86% recall for the malignant class on this particular test split.

These results describe performance on the selected dataset split; they do not establish clinical reliability.

## Key Learnings

- How SVM separates classes using decision boundaries.
- The difference between Linear and RBF kernels.
- Why feature scaling is important for SVM.
- How the `C` parameter affects the trade-off between margin size and classification errors.
- How cross-validation helps estimate model performance during tuning.
- Why accuracy alone is insufficient for evaluating classification models.
- Why recall matters when analyzing false negatives in an imbalanced or safety-sensitive classification task.

## Project Structure

```text
breast-cancer-classification-svm/
├── SVM_Breast_Cancer.ipynb
├── data.csv
└── README.md
```

## How to Run

1. Clone or download this repository.
2. Install the required Python libraries.
3. Open `SVM_Breast_Cancer.ipynb` in Jupyter Notebook or VS Code.
4. Ensure `data.csv` is available at the path expected by the notebook.
5. Run the notebook cells in order.

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Future Improvements

- Evaluate performance using repeated or stratified cross-validation.
- Compare SVM with Logistic Regression, Random Forest, and Gradient Boosting.
- Explore precision-recall curves and decision thresholds.
- Add a reproducible prediction script and model-saving workflow.
- Investigate model interpretability and the limitations of medical classification datasets.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for binary classification using Support Vector Machines. By comparing Linear and RBF kernels and tuning hyperparameters, the project achieved **97.37% test accuracy** on the selected split.

It strengthened practical understanding of preprocessing, feature scaling, model selection, cross-validation, and classification evaluation.

---

**Author:** Nagaraj  
**Domain:** Machine Learning · Classification · Scikit-learn