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
