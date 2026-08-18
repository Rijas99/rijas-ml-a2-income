from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from model.data_prep import load_data, make_preprocessor, split_data


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
        ),
    }


def calculate_metrics(model, features, target):
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    return {
        "Accuracy": accuracy_score(target, predictions),
        "AUC": roc_auc_score(target, probabilities),
        "Precision": precision_score(target, predictions, zero_division=0),
        "Recall": recall_score(target, predictions, zero_division=0),
        "F1": f1_score(target, predictions, zero_division=0),
        "MCC": matthews_corrcoef(target, predictions),
    }


def train_models(csv_path):
    data = load_data(csv_path)
    x_train, x_test, y_train, y_test = split_data(data)
    trained_models = {}
    result_rows = []

    for model_name, classifier in get_models().items():
        model = Pipeline(
            [
                ("preprocessing", make_preprocessor(x_train)),
                ("classifier", classifier),
            ]
        )
        model.fit(x_train, y_train)

        model_result = {"ML Model Name": model_name}
        model_result.update(calculate_metrics(model, x_test, y_test))
        result_rows.append(model_result)
        trained_models[model_name] = model

    results = pd.DataFrame(result_rows)
    test_split = (x_test, y_test)
    return trained_models, results, test_split


if __name__ == "__main__":
    project_folder = Path(__file__).resolve().parents[1]
    _, model_results, _ = train_models(project_folder / "adult.csv")
    print(model_results.round(4).to_string(index=False))
