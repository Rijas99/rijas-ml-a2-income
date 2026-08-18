from io import BytesIO
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT_FOLDER = Path(__file__).resolve().parent
MODEL_FOLDER = ROOT_FOLDER / "model" / "saved_models"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}

METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


st.set_page_config(
    page_title="Income Model Lab",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #14213d;
        --coral: #e07a5f;
        --sand: #f4f1de;
        --paper: #fffdf7;
        --sage: #81b29a;
        --gold: #f2cc8f;
    }
    .stApp {
        background: var(--sand);
        color: var(--ink);
    }
    [data-testid="stHeader"] {
        background: rgba(244, 241, 222, 0.82);
    }
    .block-container {
        max-width: 1240px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .income-hero {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 2rem;
        padding: 2.2rem 2.4rem;
        margin-bottom: 1.6rem;
        border-radius: 28px;
        background: var(--ink);
        color: var(--paper);
        box-shadow: 0 16px 40px rgba(20, 33, 61, 0.16);
    }
    .income-hero h1 {
        margin: 0.3rem 0 0.65rem;
        font-size: clamp(2.2rem, 5vw, 4.5rem);
        line-height: 0.98;
        letter-spacing: -0.05em;
    }
    .income-hero p {
        max-width: 670px;
        margin: 0;
        color: #d9deea;
        font-size: 1rem;
    }
    .eyebrow {
        color: var(--gold);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.17em;
    }
    .hero-stamp {
        min-width: 180px;
        padding: 1rem 1.1rem;
        border: 1px solid rgba(255,255,255,0.28);
        border-radius: 18px;
        color: var(--gold);
        font-weight: 700;
        text-align: center;
    }
    .step-title {
        margin: 0 0 0.25rem;
        color: var(--coral);
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }
    .panel-heading {
        margin: 0 0 0.8rem;
        color: var(--ink);
        font-size: 1.35rem;
        font-weight: 750;
    }
    .section-heading {
        margin: 2rem 0 0.2rem;
        color: var(--ink);
        font-size: 1.65rem;
        font-weight: 800;
    }
    .section-copy {
        margin-bottom: 1.1rem;
        color: #5d6578;
    }
    .metric-card {
        min-height: 112px;
        padding: 1.15rem 1rem;
        border: 1px solid rgba(20, 33, 61, 0.10);
        border-top: 5px solid var(--coral);
        border-radius: 18px;
        background: var(--paper);
        box-shadow: 0 8px 22px rgba(20, 33, 61, 0.07);
    }
    .metric-value {
        color: var(--ink);
        font-size: 1.8rem;
        font-weight: 850;
        line-height: 1;
    }
    .metric-label {
        margin-top: 0.55rem;
        color: #687084;
        font-size: 0.72rem;
        font-weight: 750;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .data-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin: 0.2rem 0 1.1rem;
    }
    .data-pill {
        padding: 0.45rem 0.75rem;
        border-radius: 999px;
        background: #e4efe9;
        color: #315c4d;
        font-size: 0.78rem;
        font-weight: 700;
    }
    div[data-testid="stFileUploader"] section {
        border: 1.5px dashed var(--coral);
        border-radius: 18px;
        background: rgba(255, 253, 247, 0.72);
    }
    div[data-testid="stFileUploader"] button {
        border: 0;
        background: var(--ink);
        color: var(--paper);
    }
    .stDownloadButton button {
        border: 0;
        background: var(--coral);
        color: #ffffff;
        font-weight: 750;
    }
    .stDownloadButton button:hover {
        background: #c9664f;
        color: #ffffff;
    }
    div[data-baseweb="select"] > div {
        border-color: rgba(20, 33, 61, 0.22);
        border-radius: 14px;
        background: var(--paper);
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(20, 33, 61, 0.10);
        border-radius: 16px;
        overflow: hidden;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        background: rgba(255, 253, 247, 0.8);
        padding: 0.4rem 1rem;
    }
    #MainMenu, footer, [data-testid="stToolbar"] {
        visibility: hidden;
    }
    @media (max-width: 760px) {
        .income-hero {
            align-items: flex-start;
            flex-direction: column;
            padding: 1.6rem;
        }
        .hero-stamp {
            width: 100%;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    return {
        name: joblib.load(MODEL_FOLDER / file_name)
        for name, file_name in MODEL_FILES.items()
    }


@st.cache_data
def load_saved_results():
    return pd.read_csv(ROOT_FOLDER / "model_results.csv")


def read_target(values):
    if pd.api.types.is_numeric_dtype(values):
        target = pd.to_numeric(values, errors="coerce")
    else:
        labels = values.astype(str).str.strip()
        labels = labels.replace(
            {
                "<=50K": 0,
                ">50K": 1,
                "<=50K.": 0,
                ">50K.": 1,
                "0": 0,
                "1": 1,
            }
        )
        target = pd.to_numeric(labels, errors="coerce")

    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError("The income column must contain 0/1 or <=50K/>50K values")

    return target.astype(int)


def prepare_upload(file_bytes, feature_columns):
    uploaded_data = pd.read_csv(BytesIO(file_bytes))
    required_columns = set(feature_columns) | {"income"}
    missing_columns = sorted(required_columns - set(uploaded_data.columns))

    if missing_columns:
        raise ValueError("Missing columns: " + ", ".join(missing_columns))

    features = uploaded_data.loc[:, feature_columns].copy()
    text_columns = features.select_dtypes(include=["object", "string"]).columns
    features[text_columns] = features[text_columns].apply(
        lambda column: column.str.strip()
    )
    features[text_columns] = features[text_columns].replace("?", np.nan)
    target = read_target(uploaded_data["income"])
    return uploaded_data, features, target


def get_metrics(target, predictions, probabilities):
    return {
        "Accuracy": accuracy_score(target, predictions),
        "AUC": roc_auc_score(target, probabilities),
        "Precision": precision_score(target, predictions, zero_division=0),
        "Recall": recall_score(target, predictions, zero_division=0),
        "F1": f1_score(target, predictions, zero_division=0),
        "MCC": matthews_corrcoef(target, predictions),
    }


@st.cache_data(show_spinner=False)
def evaluate_upload(file_bytes):
    models = load_models()
    first_model = next(iter(models.values()))
    feature_columns = list(first_model.feature_names_in_)
    uploaded_data, features, target = prepare_upload(file_bytes, feature_columns)
    rows = []
    predictions = {}
    probabilities = {}

    for model_name, model in models.items():
        model_predictions = model.predict(features)
        model_probabilities = model.predict_proba(features)[:, 1]
        row = {"ML Model Name": model_name}
        row.update(get_metrics(target, model_predictions, model_probabilities))
        rows.append(row)
        predictions[model_name] = model_predictions
        probabilities[model_name] = model_probabilities

    comparison = pd.DataFrame(rows)
    return uploaded_data, target, comparison, predictions, probabilities


def benchmark_figure(results):
    chart_metrics = ["Accuracy", "AUC", "F1"]
    colors = ["#e07a5f", "#14213d", "#81b29a"]
    positions = np.arange(len(results))
    bar_width = 0.23
    figure, axis = plt.subplots(figsize=(8.2, 4.4))

    for offset, metric in enumerate(chart_metrics):
        axis.barh(
            positions + (offset - 1) * bar_width,
            results[metric],
            height=bar_width,
            label=metric,
            color=colors[offset],
        )

    axis.set_yticks(positions, results["ML Model Name"])
    axis.set_xlim(0.45, 1.0)
    axis.set_xlabel("Score")
    axis.grid(axis="x", alpha=0.18)
    axis.legend(frameon=False, ncols=3, loc="lower right")
    axis.set_facecolor("#fffdf7")
    figure.patch.set_facecolor("#fffdf7")
    figure.tight_layout()
    return figure


def confusion_figure(target, predictions):
    matrix = confusion_matrix(target, predictions)
    figure, axis = plt.subplots(figsize=(5.4, 4.2))
    color_map = sns.light_palette("#e07a5f", as_cmap=True)
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap=color_map,
        cbar=False,
        square=True,
        ax=axis,
    )
    axis.set_xlabel("Predicted income")
    axis.set_ylabel("Actual income")
    axis.set_xticklabels(["<=50K", ">50K"])
    axis.set_yticklabels(["<=50K", ">50K"], rotation=0)
    figure.patch.set_facecolor("#fffdf7")
    figure.tight_layout()
    return figure


def show_metric_cards(result_row):
    columns = st.columns(6)

    for column, metric in zip(columns, METRICS):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{result_row[metric]:.3f}</div>
                    <div class="metric-label">{metric}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


st.markdown(
    """
    <div class="income-hero">
        <div>
            <div class="eyebrow">ADULT CENSUS / CLASSIFICATION STUDY</div>
            <h1>Income Model Lab</h1>
            <p>Upload labelled census test data, compare five classifiers on the same rows, and inspect where each prediction succeeds or fails.</p>
        </div>
        <div class="hero-stamp">5 MODELS<br>6 METRICS</div>
    </div>
    """,
    unsafe_allow_html=True,
)

upload_column, model_column = st.columns([1.15, 0.85], gap="large")

with upload_column:
    st.markdown('<div class="step-title">Step 01</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-heading">Bring your test set</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a labelled Adult Income CSV",
        type=["csv"],
        label_visibility="collapsed",
    )
    st.caption("The CSV must include the 14 input columns and the income column.")

with model_column:
    st.markdown('<div class="step-title">Step 02</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-heading">Choose a model to inspect</div>', unsafe_allow_html=True)
    selected_model = st.selectbox(
        "Model",
        list(MODEL_FILES),
        label_visibility="collapsed",
    )
    st.download_button(
        "Download the prepared test CSV",
        data=(ROOT_FOLDER / "test_data.csv").read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
        width="stretch",
    )

if uploaded_file is None:
    saved_results = load_saved_results()
    st.markdown('<div class="section-heading">Saved benchmark</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">These scores come from the fixed 20% test split. Upload the CSV above to run the interactive comparison.</div>',
        unsafe_allow_html=True,
    )
    table_column, chart_column = st.columns([1.1, 0.9], gap="large")

    with table_column:
        st.dataframe(
            saved_results.style.format({metric: "{:.4f}" for metric in METRICS}),
            width="stretch",
            hide_index=True,
        )

    with chart_column:
        saved_chart = benchmark_figure(saved_results)
        st.pyplot(saved_chart, width="stretch")
        plt.close(saved_chart)
else:
    try:
        with st.spinner("Running the five saved models on this CSV..."):
            (
                uploaded_data,
                target,
                comparison,
                all_predictions,
                all_probabilities,
            ) = evaluate_upload(uploaded_file.getvalue())
    except Exception as error:
        st.error(f"The uploaded CSV could not be evaluated: {error}")
        st.stop()

    selected_result = comparison.loc[
        comparison["ML Model Name"] == selected_model
    ].iloc[0]
    selected_predictions = all_predictions[selected_model]
    selected_probabilities = all_probabilities[selected_model]
    positive_rate = target.mean() * 100

    st.markdown(
        f"""
        <div class="data-strip">
            <div class="data-pill">{len(uploaded_data):,} uploaded rows</div>
            <div class="data-pill">{positive_rate:.1f}% above 50K</div>
            <div class="data-pill">Detailed view: {selected_model}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-heading">Selected model scorecard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Six measures calculated directly from the uploaded rows.</div>',
        unsafe_allow_html=True,
    )
    show_metric_cards(selected_result)

    st.markdown('<div class="section-heading">All-model comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Every classifier is evaluated against the same income labels.</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        comparison.style.format({metric: "{:.4f}" for metric in METRICS}).highlight_max(
            subset=METRICS,
            color="#dcefe5",
        ),
        width="stretch",
        hide_index=True,
    )

    report_tab, matrix_tab, prediction_tab = st.tabs(
        ["Classification report", "Confusion matrix", "Prediction rows"]
    )

    with report_tab:
        report = pd.DataFrame(
            classification_report(
                target,
                selected_predictions,
                target_names=["<=50K", ">50K"],
                output_dict=True,
                zero_division=0,
            )
        ).transpose()
        st.dataframe(
            report.style.format("{:.4f}"),
            width="stretch",
        )

    with matrix_tab:
        matrix_chart = confusion_figure(target, selected_predictions)
        st.pyplot(matrix_chart, width="content")
        plt.close(matrix_chart)

    with prediction_tab:
        prediction_data = uploaded_data.copy()
        prediction_data["predicted_income"] = np.where(
            selected_predictions == 1,
            ">50K",
            "<=50K",
        )
        prediction_data["probability_above_50k"] = selected_probabilities
        st.dataframe(
            prediction_data.head(50),
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "Download predictions",
            data=prediction_data.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_model.lower().replace(' ', '_')}_predictions.csv",
            mime="text/csv",
        )
