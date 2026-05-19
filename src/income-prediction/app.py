#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = (
    'IncomePredictionApp',
)

from typing import Literal, TYPE_CHECKING

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

if TYPE_CHECKING:
    from model import IncomePredictionModel


INCOME_PREDICTION_WORKCLASS_LITERALS = Literal[
    'Federal-gov',
    'Local-gov',
    'Never-worked',
    'Private',
    'Self-emp-inc',
    'Self-emp-not-inc',
    'State-gov',
    'Without-pay',
]


INCOME_PREDICTION_EDUCATION_LITERALS = Literal[
    '10th',
    '11th',
    '12th',
    '1st-4th',
    '5th-6th',
    '7th-8th',
    '9th',
    'Assoc-acdm',
    'Assoc-voc',
    'Bachelors',
    'Doctorate',
    'HS-grad',
    'Masters',
    'Preschool',
    'Prof-school',
    'Some-college',
]


INCOME_PREDICTION_MARITALSTATUS_LITERALS = Literal[
    'Divorced',
    'Married-AF-spouse',
    'Married-civ-spouse',
    'Married-spouse-absent',
    'Never-married',
    'Separated',
    'Widowed',
]


INCOME_PREDICTION_OCCUPATION_LITERALS = Literal[
    'Adm-clerical',
    'Armed-Forces',
    'Craft-repair',
    'Exec-managerial',
    'Farming-fishing',
    'Handlers-cleaners',
    'Machine-op-inspct',
    'Other-service',
    'Priv-house-serv',
    'Prof-specialty',
    'Protective-serv',
    'Sales',
    'Tech-support',
    'Transport-moving',
]


INCOME_PREDICTION_RELATIONSHIP_LITERALS = Literal[
    'Husband',
    'Not-in-family',
    'Other-relative',
    'Own-child',
    'Unmarried',
    'Wife',
]


INCOME_PREDICTION_RACE_LITERALS = Literal[
    'Amer-Indian-Eskimo',
    'Asian-Pac-Islander',
    'Black',
    'Other',
    'White',
]


INCOME_PREDICTION_SEX_LITERALS = Literal[
    'Female',
    'Male',
]


INCOME_PREDICTION_NATIVE_COUNTRY_LITERALS = Literal[
    'Cambodia',
    'Canada',
    'China',
    'Columbia',
    'Cuba',
    'Dominican-Republic',
    'Ecuador',
    'El-Salvador',
    'England',
    'France',
    'Germany',
    'Greece',
    'Guatemala',
    'Haiti',
    'Holand-Netherlands',
    'Honduras',
    'Hong',
    'Hungary',
    'India',
    'Iran',
    'Ireland',
    'Italy',
    'Jamaica',
    'Japan',
    'Laos',
    'Mexico',
    'Nicaragua',
    'Outlying-US(Guam-USVI-etc)',
    'Peru',
    'Philippines',
    'Poland',
    'Portugal',
    'Puerto-Rico',
    'Scotland',
    'South',
    'Taiwan',
    'Thailand',
    'Trinadad&Tobago',
    'United-States',
    'Vietnam',
    'Yugoslavia',
]


INCOME_PREDICTION_NATIVE_SALARY_LITERALS = Literal[
    '<=50K',
    '>50K',
]


class IncomePredictionGreetingResponse(BaseModel):
    message: str


class IncomePredictionInferenceRequest(BaseModel):
    """Pydantic model for the input data of the prediction endpoint."""
    age:             int = Field(ge=16, le=100)
    """Age"""
    race:            INCOME_PREDICTION_RACE_LITERALS
    """Race"""
    sex:             INCOME_PREDICTION_SEX_LITERALS
    """Sex"""
    fnlwgt:          int = Field(ge=12_000, le=1_500_000)
    """Final Weight"""
    marital_status:  INCOME_PREDICTION_MARITALSTATUS_LITERALS = Field(alias='marital-status')
    """Marital Status"""
    relationship:    INCOME_PREDICTION_RELATIONSHIP_LITERALS
    """Household"""
    education:       INCOME_PREDICTION_EDUCATION_LITERALS
    """Education"""
    hours_per_week:  int = Field(alias='hours-per-week', ge=1, le=100)
    """Hours per Week"""
    capital_gain:    int = Field(alias='capital-gain', ge=0, le=100000)
    """Capital Gain"""
    capital_loss:    int = Field(alias='capital-loss', ge=0, le=50000)
    """Capital Loss"""
    native_country:  INCOME_PREDICTION_NATIVE_COUNTRY_LITERALS | SkipJsonSchema[None] = Field(alias='native-country',
                                                                                              default=None)
    """Citizenship"""
    workclass:       INCOME_PREDICTION_WORKCLASS_LITERALS | SkipJsonSchema[None] = Field(default=None)
    """Workclass"""
    occupation:      INCOME_PREDICTION_OCCUPATION_LITERALS | SkipJsonSchema[None] = Field(default=None)
    """Occupation"""


class IncomePredictionInferenceResponse(BaseModel):
    income: INCOME_PREDICTION_NATIVE_SALARY_LITERALS = Field(description='Predicted Income Class')
    """Predicted Income Class"""


class IncomePredictionExceptionResponse(BaseModel):
    detail: str = Field(description='Internal Python Exception Error Message')
    """Internal Python Exception Error Message"""


class IncomePredictionApp(FastAPI):
    """FastAPI Application serving the Income Prediction Model REST Api"""

    def __init__(self, model: 'IncomePredictionModel'):
        """Initialize the Application

        Args:
            model: pre-trained income prediction model
        """
        super().__init__(title='Income Prediction API')
        self._model = model

        # add the different routes
        self.add_api_route('/',        self.greeting, methods=['GET'])
        self.add_api_route('/predict', self.predict,  methods=['POST'],
                           responses={
                               500: {
                                   "model": IncomePredictionExceptionResponse,
                                   "description": "Internal Server Error",
                               }
                           })

    def greeting(self) -> IncomePredictionGreetingResponse:
        """Return a welcome message for the API."""
        return {'message': f'Welcome to {self.title}!'}

    def predict(self, request: IncomePredictionInferenceRequest) -> IncomePredictionInferenceResponse:
        """Predict the income class for the given input data.

        Args:
            request: input data for prediction

        Returns:
            dict: predicted income class
        """
        try:
            # convert to dataframe from record
            df = pd.DataFrame([request.model_dump(by_alias=True)])
            # run inference
            result = self._model.predict(df)
            # return result
            return {'income': result[0]}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
