#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for the Income Prediction Model training and inference flow.

We use a small synthetic dataset to verify that the model can:
- be trained and return metrics,
- be trained and predict expected values,
- be trained and overfit the tiny training split as a basic sanity check

Notice that the synthetic dataset does not contain all feature columns as an
additional robustness test.
"""

from importlib.resources import files

import pandas as pd
import pytest

from model import IncomePredictionModel
import resources


RFC_CONFIG = {
    'n_estimators': 25,
    'max_depth':    5,
    'max_features': 'sqrt',
    'random_state': 0x01234567,
}
"""Configuration for Random Forest Classifier Model"""


@pytest.fixture(scope='module')
def synthetic_data():
    data = pd.read_csv(files(resources) / 'synthetic.csv')
    assert len(data) >= 12, 'Not enough synthetic data for train/test split'
    assert len(data) <= 24, 'Too much synthetic data for sanity check'
    assert 'salary' in data.columns, 'Missing target column "salary" in synthetic data'
    return data


@pytest.fixture(scope='module')
def test_df(synthetic_data: pd.DataFrame):
    # 2/3 is training, 1/3 is test
    return synthetic_data.iloc[2 * len(synthetic_data) // 3:]


@pytest.fixture(scope='module')
def train_df(synthetic_data: pd.DataFrame):
    # 2/3 is training, 1/3 is test
    return synthetic_data.iloc[:2 * len(synthetic_data) // 3]


def test_model_train_and_validate(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Test Basic Training and Validation of the Model

    Args:
        train_df: DataFrame containing the training data
        test_df: DataFrame containing the test data
    """
    model = IncomePredictionModel.from_config(RFC_CONFIG)
    model.train(train_df)
    precision, recall, f_beta = model.validate(test_df)

    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= f_beta <= 1.0


def test_model_predict(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Test Basic Prediction of Trained Model

    Args:
        train_df: DataFrame containing the training data
        test_df: DataFrame containing the test data
    """
    model = IncomePredictionModel.from_config(RFC_CONFIG)
    model.train(train_df)
    preds = model.predict(test_df)

    assert len(preds) == len(test_df)
    assert set(preds).issubset({'<=50K', '>50K'})


def test_model_overfit(train_df: pd.DataFrame):
    """Test Overfitting of the Model

    Synthetic Test Data is small enough that RFC should be able to simply
    memorize it all if it's not completely broken.

    Args:
        train_df: DataFrame containing the training data
    """
    model = IncomePredictionModel.from_config(RFC_CONFIG)
    model.train(train_df)
    precision, recall, f_beta = model.validate(train_df)

    assert 1.0 == precision
    assert 1.0 == recall
    assert 1.0 == f_beta
