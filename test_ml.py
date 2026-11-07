import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.data import apply_label, process_data
from ml.model import compute_model_metrics, inference, train_model

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, "data", "census.csv")

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def data():
    """Load the census data once for all tests."""
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def split_data(data):
    """Split the data into train and test sets."""
    return train_test_split(
        data, test_size=0.20, random_state=42, stratify=data["salary"]
    )


@pytest.fixture(scope="module")
def processed(split_data):
    """Process train and test data the same way train_model.py does."""
    train, test = split_data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    return X_train, y_train, X_test, y_test


def test_train_test_split_size(data, split_data):
    """
    Test that the train and test datasets have the expected size
    (80% / 20%) and that no rows are lost when splitting.
    """
    train, test = split_data
    assert len(train) + len(test) == len(data)
    assert len(test) == pytest.approx(0.20 * len(data), abs=1)
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)


def test_train_model_algorithm(processed):
    """
    Test that train_model returns a fitted RandomForestClassifier.
    """
    X_train, y_train, _, _ = processed
    model = train_model(X_train[:2000], y_train[:2000])
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "estimators_")


def test_inference_output(processed):
    """
    Test that inference returns a numpy array with one 0/1
    prediction for every row of input data.
    """
    X_train, y_train, X_test, _ = processed
    model = train_model(X_train[:2000], y_train[:2000])
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_values():
    """
    Test that compute_model_metrics returns the expected precision,
    recall and F1 on a small hand-checked example.
    """
    y = np.array([1, 1, 0, 0])
    preds = np.array([1, 0, 1, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(0.5)


def test_apply_label():
    """
    Test that apply_label turns 1 and 0 into the salary labels.
    """
    assert apply_label([1]) == ">50K"
    assert apply_label([0]) == "<=50K"
