from pathlib import Path

import joblib

from model.train_models import train_models


MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}


def save_results(project_folder):
    project_folder = Path(project_folder)
    models, results, test_split = train_models(project_folder / "adult.csv")
    x_test, y_test = test_split

    saved_model_folder = project_folder / "model" / "saved_models"
    saved_model_folder.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        model_path = saved_model_folder / MODEL_FILES[model_name]
        joblib.dump(model, model_path, compress=3)

    results.to_csv(
        project_folder / "model_results.csv",
        index=False,
        float_format="%.6f",
    )

    test_data = x_test.copy()
    test_data["income"] = y_test
    test_data.reset_index(drop=True).to_csv(project_folder / "test_data.csv", index=False)

    return saved_model_folder, results, test_data


if __name__ == "__main__":
    root_folder = Path(__file__).resolve().parents[1]
    model_folder, model_results, saved_test_data = save_results(root_folder)

    print("Saved models in:", model_folder)
    print("Saved result rows:", len(model_results))
    print("Saved test rows:", len(saved_test_data))
