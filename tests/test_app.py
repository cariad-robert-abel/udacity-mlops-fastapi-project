#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for the Income Prediction Model FastAPI Application

We use a small pre-trained model that does not perform as well as the production
model, but well enough for our unit tests.

We quickly check each endpoint for each expected outcome once using the built-in
FastAPI test client.
"""

from importlib.resources import files

import pytest
from fastapi.testclient import TestClient

from app import IncomePredictionApp
from model import IncomePredictionModel
import resources


@pytest.fixture(scope='module')
def model():
    return IncomePredictionModel.from_model(files(resources) / 'model.joblib')


@pytest.fixture(scope='module')
def client(model: IncomePredictionModel):
    return TestClient(IncomePredictionApp(model))


def test_app_greeting(client: TestClient):
    """Test the Greeting Endpoint of the FastAPI App

    Args:
        client: FastAPI Test Client for the Income Prediction App
    """
    response = client.get('/')
    assert response.status_code == 200, 'greeting endpoint failed'
    assert 'message' in response.json(), 'message field missing from greeting response'
    assert response.json()['message'].startswith('Welcome to'), 'greeting message incorrect'


def test_app_predict_le50k(client: TestClient):
    """Test the Prediction Endpoint of the FastAPI App <=50K

    Args:
        client: FastAPI Test Client for the Income Prediction App
    """
    # line 1 from census data CSV
    request = {
        "age": 39,
        "workclass": "State-gov",
        "fnlwgt": 77516,
        "education": "Bachelors",
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    response = client.post('/predict', json=request)
    assert response.status_code == 200, 'greeting endpoint failed'
    assert 'income' in response.json(), 'income field missing from predict response'
    assert response.json()['income'] == '<=50K', 'prediction incorrect'


def test_app_predict_gt50k(client: TestClient):
    """Test the Prediction Endpoint of the FastAPI App >50K

    Args:
        client: FastAPI Test Client for the Income Prediction App
    """
    # line 12 from census data CSV
    request = {
        "age": 37,
        "workclass": "Private",
        "fnlwgt": 280464,
        "education": "Some-college",
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "Black",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 80,
        "native-country": "United-States"
    }
    response = client.post('/predict', json=request)
    assert response.status_code == 200, 'greeting endpoint failed'
    assert 'income' in response.json(), 'income field missing from predict response'
    assert response.json()['income'] == '>50K', 'prediction incorrect'
