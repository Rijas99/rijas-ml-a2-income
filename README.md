# Adult Income Classification

This project is for Machine Learning Assignment 2. The aim is to predict whether a person's income is above 50K using census details from the Adult Income dataset.

The classification models and Streamlit dashboard will be added step by step.

## Data preparation

The `?` values in the dataset are treated as missing values and duplicate rows are removed. The data is divided into 80% training and 20% testing data using a stratified split with random state 42.

## Models added so far

- Logistic Regression
- Decision Tree
- k-Nearest Neighbour
- Gaussian Naive Bayes
- Random Forest

All models use the same preprocessing steps and test split so their results can be compared fairly.

## Model comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8477 | 0.9020 | 0.7258 | 0.5848 | 0.6477 | 0.5572 |
| Decision Tree | 0.8520 | 0.8923 | 0.7406 | 0.5878 | 0.6554 | 0.5688 |
| kNN | 0.8338 | 0.8741 | 0.6700 | 0.6023 | 0.6344 | 0.5284 |
| Naive Bayes | 0.6173 | 0.8329 | 0.3777 | 0.9242 | 0.5363 | 0.3855 |
| Random Forest | 0.8588 | 0.9148 | 0.7788 | 0.5728 | 0.6601 | 0.5844 |

In `test_data.csv`, income values at or below 50K are stored as 0 and values above 50K are stored as 1.
