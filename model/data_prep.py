from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "income"


def load_data(csv_path):
    data = pd.read_csv(csv_path)
    text_columns = data.select_dtypes(include=["object", "string"]).columns
    data[text_columns] = data[text_columns].apply(lambda column: column.str.strip())
    data[text_columns] = data[text_columns].replace("?", np.nan)
    data = data.drop_duplicates().reset_index(drop=True)

    income_values = {"<=50K": 0, ">50K": 1}
    data[TARGET_COLUMN] = data[TARGET_COLUMN].map(income_values)

    if data[TARGET_COLUMN].isna().any():
        raise ValueError("Unexpected value found in the income column")

    return data


def split_data(data, test_size=0.2, random_state=42):
    features = data.drop(columns=TARGET_COLUMN)
    target = data[TARGET_COLUMN]

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def make_preprocessor(features):
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    category_columns = features.select_dtypes(exclude="number").columns.tolist()

    numeric_steps = Pipeline(
        [
            ("fill_missing", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    category_steps = Pipeline(
        [
            ("fill_missing", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_steps, numeric_columns),
            ("category", category_steps, category_columns),
        ]
    )


if __name__ == "__main__":
    project_folder = Path(__file__).resolve().parents[1]
    cleaned_data = load_data(project_folder / "adult.csv")
    x_train, x_test, y_train, y_test = split_data(cleaned_data)

    print("Rows after cleaning:", len(cleaned_data))
    print("Training rows:", len(x_train))
    print("Testing rows:", len(x_test))
    print("Training target counts:")
    print(y_train.value_counts().sort_index())
