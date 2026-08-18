from pathlib import Path
import unittest

import joblib
import pandas as pd

from model.data_prep import load_data, split_data
from model.save_results import MODEL_FILES


ROOT_FOLDER = Path(__file__).resolve().parents[1]


class ProjectChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data = pd.read_csv(ROOT_FOLDER / "test_data.csv")
        cls.results = pd.read_csv(ROOT_FOLDER / "model_results.csv")

    def test_clean_data_and_split_sizes(self):
        data = load_data(ROOT_FOLDER / "adult.csv")
        x_train, x_test, y_train, y_test = split_data(data)

        self.assertEqual(len(data), 48790)
        self.assertEqual(len(x_train), 39032)
        self.assertEqual(len(x_test), 9758)
        self.assertEqual(set(y_train.unique()), {0, 1})
        self.assertEqual(set(y_test.unique()), {0, 1})

    def test_model_results_table(self):
        expected_columns = {
            "ML Model Name",
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1",
            "MCC",
        }

        self.assertEqual(len(self.results), 5)
        self.assertEqual(set(self.results.columns), expected_columns)
        self.assertEqual(set(self.results["ML Model Name"]), set(MODEL_FILES))

    def test_saved_models_make_predictions(self):
        features = self.test_data.drop(columns="income").head(5)

        for model_name, file_name in MODEL_FILES.items():
            model_path = ROOT_FOLDER / "model" / "saved_models" / file_name
            model = joblib.load(model_path)
            predictions = model.predict(features)
            probabilities = model.predict_proba(features)

            self.assertEqual(len(predictions), 5, model_name)
            self.assertEqual(probabilities.shape, (5, 2), model_name)


if __name__ == "__main__":
    unittest.main()
