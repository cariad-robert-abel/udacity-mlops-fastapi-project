#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = (
    'IncomePredictionModel',
)

import logging
import sys
from typing import Any, Self, TYPE_CHECKING

import numpy as np
import joblib
import sklearn

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder

if TYPE_CHECKING:
    import os

    import numpy.typing as npt
    import pandas as pd

logger = logging.getLogger(__name__)


class IncomePredictionModel:

    __create_key = object()
    """Token to prevent direct instantiation"""

    def __init__(self, pipeline: Pipeline, *, key: object):
        """Create an Income Prediction Model

        Private, use the from_config()/from_model() methods to create an instance.
        """
        if key is not IncomePredictionModel.__create_key:
            raise ValueError("Use the from_config()/from_model() methods to create an instance.")

        self._pipeline = pipeline

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'Self':
        """Create an IncomePredictionModel from a configuration dictionary
           ready to be trained.

        Args:
            config: A dictionary containing the configuration parameters for Random Forest Classifier.

        Returns:
            An instance of IncomePredictionModel initialized with the given configuration.
        """
        # transform numeric data and impute with zeros
        # frontend needs to deal with mandatory values
        numeric_xform = make_pipeline(
            SimpleImputer(strategy='constant', fill_value=0)
        )
        # transform categorical data and impute missing values with
        category_xform = make_pipeline(
            SimpleImputer(strategy='constant', fill_value='<Unknown>'),
            OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        )

        # create the column transformer
        col_xform = ColumnTransformer(
            transformers=[
                ('numeric',     numeric_xform, make_column_selector(dtype_include=np.number)),
                ('categorical', category_xform, make_column_selector(dtype_include=['object', 'string'])),
            ],
            # drop untransformed columns
            remainder='drop'
        )

        # create the pipeline
        pipeline = Pipeline(
            steps=[
                ('preprocessor', col_xform),
                ('classifier', RandomForestClassifier(**config))
            ]
        )

        # construct the model with out skeleton pipeline
        return cls(pipeline, key=cls.__create_key)

    @classmethod
    def from_model(cls, filename: 'str | os.PathLike[str]') -> 'Self':
        """Create an IncomePredictionModel from a pre-trained model.

        Args:
            filename: path to the model file

        Returns:
            An instance of IncomePredictionModel initialized with the given model.
        """
        model = joblib.load(filename)

        # make sure it's a dictionary
        if not isinstance(model, dict):
            raise ValueError(f"Expected model file to contain a dictionary, got {type(model)}")

        # make sure mandatory keys are present
        keys = ('pipeline', 'sklearn-version', 'python-version')
        missing_keys = tuple(x for x in keys if x not in model)
        if missing_keys:
            raise ValueError(
                f'Expected model file to contain keys {", ".join(keys)}, but missing key(s) {", ".join(missing_keys)}'
            )

        logger.info(
            f'Loaded model from {filename}: scikit-learn {model["sklearn-version"]} (Python {model["python-version"]})'
        )

        # construct the model from a fixed pre-trained pipeline
        return cls(model['pipeline'], key=cls.__create_key)

    def store_model(self, filename: 'str | os.PathLike[str]'):
        """Store the current IncomePredictionModel to a file.

        Args:
            filename: path to the model file
        """
        model = {
            'pipeline': self._pipeline,
            'sklearn-version': sklearn.__version__,
            'python-version': f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}',
        }
        joblib.dump(model, filename)
        logger.info(f'Model stored to {filename}')

    @staticmethod
    def _prepare_data(
        data: 'pd.DataFrame',
        label: str | None = None,
        weights: str | None = None,
    ) -> tuple['pd.DataFrame', 'pd.Series', 'pd.Series']:
        """Prepare the dataset for training or inference.

        Args:
            data: Dataset to prepare
            label: Column name for the target variable
            weights: Column name for the sample weights

        Returns:
            Tuple (X, y, w) containing the prepared features, labels, and sample weights.
        """
        X = data.drop(columns=[label, weights], errors='ignore')
        y = data[label] if label else None
        w = data[weights] if weights else None
        return X, y, w

    def train(self, train: 'pd.DataFrame'):
        """Train the Random Forest Classifier

        Trains a machine learning model and saves it to the `model` directory.

        Args:
            train: Training Dataset
        """
        X_train, y_train, w_train = IncomePredictionModel._prepare_data(train, label='salary', weights='fnlwgt')

        logger.info(f'Training model on {len(X_train)} samples...')
        self._pipeline.fit(X_train, y_train, classifier__sample_weight=w_train)

    def validate(self, test: 'pd.DataFrame') -> tuple[float, float, float]:
        """Validates the trained machine learning model using precision, recall, and F1.

        Args:
            test: Test Dataset

        Returns:
            A tuple containing the precision, recall, and F1 score of the trained model on the test set.
        """
        X, y, w = IncomePredictionModel._prepare_data(test, label='salary', weights='fnlwgt')
        # calculate predictions
        preds = self._pipeline.predict(X)
        # keyword-arguments for score functions
        kwargs = {
            # scikit-learn stores labels in lexicographic ascending order, so outputs will
            # be stable for same class labels ('<=50K' will sort before '>50K').
            # See https://scikit-learn.org/stable/glossary.html#term-classes_
            # Therefore, use top-most classifier label as positive class
            'pos_label': self._pipeline.named_steps['classifier'].classes_[-1],
            'sample_weight': w,
            # treat zero-division as perfect score
            'zero_division': 1,
        }
        # calculate weighed metrics against known labels
        fbeta = fbeta_score(y, preds, beta=1, **kwargs)
        precision = precision_score(y, preds, **kwargs)
        recall = recall_score(y, preds, **kwargs)
        return precision, recall, fbeta

    def predict(self, data: 'pd.DataFrame') -> 'npt.NDArray[np.str_]':
        """ Run model inferences and return the predictions.

        Args:
            data: Dataframe containing the features to run inference on.

        Returns:
            A numpy array containing the predicted labels.
        """
        X, _, _ = IncomePredictionModel._prepare_data(data, label='salary', weights='fnlwgt')
        return self._pipeline.predict(X)
