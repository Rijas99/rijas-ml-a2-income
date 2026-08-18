# Adult Income Classification

This is my Machine Learning Assignment 2 project. I used the Adult Census Income dataset to predict whether a person's yearly income is above 50K. I trained five classification models, compared their results, and built a Streamlit app where the saved models can be tested.

## a. Problem statement

The aim of this project is to predict whether a person's annual income is `<=50K` or `>50K` using census details such as age, education, occupation, working hours, and capital gain or loss.

There are only two possible output classes, so this is a binary classification problem. In the prepared data:

- `0` means the income is less than or equal to 50K.
- `1` means the income is greater than 50K.

## b. Dataset description

I used the public [Adult Census Income dataset](https://archive.ics.uci.edu/dataset/2/adult) from the UCI Machine Learning Repository. The dataset used in this project is stored in `adult.csv`.

- Original rows: 48,842
- Input features: 14
- Numerical features: 6
- Categorical features: 8
- Target column: `income`
- `<=50K` records: 37,155
- `>50K` records: 11,687

The dataset contains `?` in the `workclass`, `occupation`, and `native-country` columns. I treated these as missing values. I also removed 52 duplicate rows, which left 48,790 rows for the experiment.

For numerical columns, I filled missing values using the median and then applied standard scaling. For categorical columns, I filled missing values using the most frequent value and used one-hot encoding.

The income classes are not balanced because around 76% of the records belong to the `<=50K` class. Because of this, I compared the models using six metrics instead of depending only on accuracy.

## Project workflow

The main steps I followed were:

1. Load the dataset and check its columns and target values.
2. Replace the `?` values with missing values and remove duplicate rows.
3. Separate the input features and the income target.
4. Split the data into 80% training data and 20% testing data.
5. Use stratified sampling so both splits have a similar class distribution.
6. Preprocess the numerical and categorical columns.
7. Train and evaluate five classification models on the same test data.
8. Save the trained pipelines and build the Streamlit app.

I used `random_state=42` for the split and for models that support it, so the results can be reproduced.

## c. GitHub repository link

[Rijas99/rijas-ml-a2-income](https://github.com/Rijas99/rijas-ml-a2-income)

## Live Streamlit app

[Income Model Lab](https://rijas-income-model-lab.streamlit.app)

## d. Models used

I trained the following five classification models:

1. Logistic Regression
2. Decision Tree Classifier
3. k-Nearest Neighbour Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier

All five models use the same preprocessing steps, training data, and testing data so that the comparison is fair.

## Model comparison

The following results were obtained from the fixed 20% test split:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8477 | 0.9020 | 0.7258 | 0.5848 | 0.6477 | 0.5572 |
| Decision Tree | 0.8520 | 0.8923 | 0.7406 | 0.5878 | 0.6554 | 0.5688 |
| kNN | 0.8338 | 0.8741 | 0.6700 | 0.6023 | 0.6344 | 0.5284 |
| Naive Bayes | 0.6173 | 0.8329 | 0.3777 | **0.9242** | 0.5363 | 0.3855 |
| Random Forest | **0.8588** | **0.9148** | **0.7788** | 0.5728 | **0.6601** | **0.5844** |

## What I observed

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | This model gave a strong starting result. Its AUC was the second highest, and it gave a good balance between accuracy and precision. Its lower recall means it missed some people earning above 50K. |
| Decision Tree | The Decision Tree gave slightly better accuracy, F1, and MCC than Logistic Regression. Its AUC was lower, so its probability ranking was not as strong. |
| kNN | kNN gave reasonable recall, but its accuracy, AUC, and MCC were lower than Logistic Regression and Decision Tree. The large number of columns created by one-hot encoding may make distance comparison more difficult. |
| Naive Bayes | Naive Bayes had the highest recall at 0.9242 and found most higher-income records. Its low precision shows that it also produced many false positive predictions. |
| Random Forest | Random Forest gave the best accuracy, AUC, precision, F1, and MCC. It did not have the highest recall, but it gave the most balanced overall result. |
| Overall Winner | I selected **Random Forest** because it had the best score in five of the six metrics. MCC was useful here because the income classes are imbalanced. |

## Streamlit app

I created the Streamlit app to make the model results easier to check. The app allows the user to:

- Upload a labelled Adult Income CSV file.
- Download the prepared `test_data.csv` file.
- Evaluate all five saved models on the uploaded rows.
- Select one model for a detailed view.
- View Accuracy, AUC, Precision, Recall, F1, and MCC.
- View the classification report and confusion matrix.
- Preview and download the predictions.

The light-coloured cells in the comparison table show the best score in each metric. Random Forest leads five metrics, while Naive Bayes has the highest recall.

## Project structure

```text
rijas-ml-a2-income/
|-- .streamlit/
|   `-- config.toml
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

## How to run the project

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the environment

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Start the Streamlit app

```bash
streamlit run app.py
```

The app normally opens automatically. It can also be opened at `http://localhost:8501`.

## Train the models again

```bash
python -m model.train_models
python -m model.save_results
```

The second command saves the trained model pipelines, recreates `model_results.csv`, and creates `test_data.csv`.

## Run the tests

```bash
python -m unittest discover -s tests -v
```

## Tools and libraries

I used Python, Pandas, NumPy, Scikit-learn, Streamlit, Matplotlib, Seaborn, and Joblib in this project.

## Conclusion

This project helped me understand why an imbalanced classification problem should not be judged using accuracy alone. Naive Bayes showed how a model can have very high recall but still produce many incorrect positive predictions. Random Forest gave the strongest balanced result across the six metrics.

I also learned how to keep preprocessing and the classifier together in one saved pipeline. This made it easier to use the same transformations when new CSV data was uploaded in the Streamlit app.
