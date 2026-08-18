# Adult Income Classification

This project compares five machine learning classification models using the Adult Census Income dataset. A Streamlit app is included to upload labelled test data, compare all model scores, and inspect one selected model in more detail.

## Problem statement

The aim is to predict whether a person's annual income is above 50K from census details such as age, education, occupation, working hours, and capital gain or loss. This is a binary classification problem:

- `0` means income is less than or equal to 50K.
- `1` means income is greater than 50K.

## Dataset description

The project uses the public [Adult Census Income dataset](https://archive.ics.uci.edu/dataset/2/adult) from the UCI Machine Learning Repository.

- Original rows: 48,842
- Input features: 14
- Numerical features: 6
- Categorical features: 8
- Target column: `income`
- `<=50K` records: 37,155
- `>50K` records: 11,687

The original data contains `?` markers in `workclass`, `occupation`, and `native-country`. These are treated as missing values. A total of 52 duplicate rows are removed, leaving 48,790 rows for the experiment.

Numerical missing values are filled using the median and then scaled. Categorical missing values are filled using the most frequent value and converted using one-hot encoding. The data is divided into 80% training and 20% testing data with a stratified split and random state 42.

## GitHub repository

[https://github.com/Rijas99/rijas-ml-a2-income](https://github.com/Rijas99/rijas-ml-a2-income)

## Live Streamlit app

The public Streamlit link will be added after deployment.

## Models used

1. Logistic Regression
2. Decision Tree Classifier
3. k-Nearest Neighbour Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier

All models use the same training data, preprocessing steps, and test data.

## Model comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8477 | 0.9020 | 0.7258 | 0.5848 | 0.6477 | 0.5572 |
| Decision Tree | 0.8520 | 0.8923 | 0.7406 | 0.5878 | 0.6554 | 0.5688 |
| kNN | 0.8338 | 0.8741 | 0.6700 | 0.6023 | 0.6344 | 0.5284 |
| Naive Bayes | 0.6173 | 0.8329 | 0.3777 | 0.9242 | 0.5363 | 0.3855 |
| Random Forest | 0.8588 | 0.9148 | 0.7788 | 0.5728 | 0.6601 | 0.5844 |

## Model observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | This is a strong baseline model. It has the second highest AUC and gives a good balance between accuracy and precision, but it misses some of the higher-income records. |
| Decision Tree | The tree gives slightly better accuracy, F1, and MCC than Logistic Regression. Its AUC is lower, showing that its probability ranking is not as strong. |
| kNN | kNN gives reasonable recall but lower accuracy, AUC, and MCC than Logistic Regression and Decision Tree. Distance calculations also make it slower on larger test files. |
| Naive Bayes | Naive Bayes has the highest recall at 0.9242, so it identifies most of the higher-income records. Its low precision and accuracy show that it also produces many false positives. |
| Random Forest | Random Forest has the best accuracy, AUC, precision, F1, and MCC. It gives the most reliable overall performance for this dataset. |
| Overall winner | **Random Forest** is the overall winner because it leads five of the six evaluation metrics. |

## Streamlit app features

- Upload a labelled Adult Income CSV file.
- Download the prepared `test_data.csv` file.
- Evaluate all five saved models on the uploaded data.
- Select one model for detailed analysis.
- View Accuracy, AUC, Precision, Recall, F1, and MCC.
- View a classification report and confusion matrix.
- Preview and download model predictions.

## Project structure

```text
rijas-ml-a2-income/
|-- app.py
|-- adult.csv
|-- test_data.csv
|-- model_results.csv
|-- requirements.txt
|-- README.md
|-- model/
|   |-- data_prep.py
|   |-- train_models.py
|   |-- save_results.py
|   `-- saved_models/
|       |-- logistic_regression.joblib
|       |-- decision_tree.joblib
|       |-- knn.joblib
|       |-- naive_bayes.joblib
|       `-- random_forest.joblib
`-- tests/
    `-- test_project.py
```

## Run the project

Create a virtual environment and install the packages:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Linux or macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` if the browser does not open automatically.

## Train the models again

```bash
python -m model.train_models
python -m model.save_results
```

The second command recreates the saved model files, `model_results.csv`, and `test_data.csv`.

## Run the checks

```bash
python -m unittest discover -s tests -v
```

## Deployment

The app is prepared for Streamlit Community Cloud:

1. Sign in to Streamlit Community Cloud using GitHub.
2. Select the `Rijas99/rijas-ml-a2-income` repository.
3. Choose the `main` branch.
4. Set the application file to `app.py`.
5. Deploy the application and copy the public app URL into this README.

For the assignment submission, the final PDF must also include the GitHub link, live app link, README content, and one execution screenshot from the BITS Virtual Lab.
