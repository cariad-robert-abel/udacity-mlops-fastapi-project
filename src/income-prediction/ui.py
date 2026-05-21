#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = (
    'IncomePredictionUI',
)

import logging
import random
from functools import partial
from typing import TYPE_CHECKING

import gradio as gr
import pandas as pd

if TYPE_CHECKING:
    from model import IncomePredictionModel


INCOME_PREDICTION_UI_ELEMENTS = [
    # ---- ROW 1 ---------------------------------------------------------------
    {
        'age': {
            'name': 'Age',
            'type': 'number',
            'min': 16,
            'max': 100,
        },
        'race': {
            'name': 'Race',
            'type': 'dropdown',
            'nillable': False,
            'values': {
                'Asian':            'Asian-Pac-Islander',
                'Black':            'Black',
                'Native American':  'Amer-Indian-Eskimo',
                'Pacific Islander': 'Asian-Pac-Islander',
                'White':            'White',
                'Other':            'Other',
            },
        },
        'sex': {
            'name': 'Sex',
            'type': 'dropdown',
            'nillable': False,
            'values': {
                'Female': 'Female',
                'Male':   'Male',
            },
        },
        'native-country': {
            'name': 'Citizenship',
            'type': 'dropdown',
            'nillable': True,
            'values': {
                'Cambodia':            'Cambodia',
                'Canada':              'Canada',
                'China':               'China',
                'Columbia':            'Columbia',
                'Cuba':                'Cuba',
                'Dominican Republic':  'Dominican-Republic',
                'Ecuador':             'Ecuador',
                'El Salvador':         'El-Salvador',
                'England':             'England',
                'France':              'France',
                'Germany':             'Germany',
                'Greece':              'Greece',
                'Guatemala':           'Guatemala',
                'Haiti':               'Haiti',
                'Netherlands':         'Holand-Netherlands',
                'Honduras':            'Honduras',
                'Hong Kong':           'Hong',
                'Hungary':             'Hungary',
                'India':               'India',
                'Iran':                'Iran',
                'Ireland':             'Ireland',
                'Italy':               'Italy',
                'Jamaica':             'Jamaica',
                'Japan':               'Japan',
                'Laos':                'Laos',
                'Mexico':              'Mexico',
                'Nicaragua':           'Nicaragua',
                'Peru':                'Peru',
                'Philippines':         'Philippines',
                'Poland':              'Poland',
                'Portugal':            'Portugal',
                'Puerto Rico':         'Puerto-Rico',
                'Scotland':            'Scotland',
                'South Korea':         'South',
                'Taiwan':              'Taiwan',
                'Thailand':            'Thailand',
                'Trinidad and Tobago': 'Trinadad&Tobago',
                'United States':       'United-States',
                'U.S. Territories and Outlying Areas': 'Outlying-US(Guam-USVI-etc)',
                'Vietnam':             'Vietnam',
                'Yugoslavia':          'Yugoslavia',
            },
        },
        'fnlwgt': {
            'name': 'Final Weight',
            'type': 'number',
            'min': 12_000,
            'max': 1_500_000,
            'randomize': True,
        },
    },
    # ---- ROW 2 ---------------------------------------------------------------
    {
        'marital-status': {
            'name': 'Marital Status',
            'type': 'dropdown',
            'nillable': False,
            'values': {
                'Never Married':              'Never-married',
                'Married (Civilian Spouse)':  'Married-civ-spouse',
                'Married (Air Force Spouse)': 'Married-AF-spouse',
                'Married (Spouse Absent)':    'Married-spouse-absent',
                'Divorced':                   'Divorced',
                'Separated':                  'Separated',
                'Widowed':                    'Widowed',
            },
        },
        'relationship': {
            'name': 'Household Role',
            'type': 'dropdown',
            'nillable': False,
            'values': {
                'Husband':        'Husband',
                'Wife':           'Wife',
                'Not in Family':  'Not-in-family',
                'Other Relative': 'Other-relative',
                'Own Child':      'Own-child',
                'Unmarried':      'Unmarried',
            },
        },
        'education': {
            'name': 'Education',
            'type': 'dropdown',
            'nillable': False,
            'values': {
                'Preschool':                          'Preschool',
                '1st to 4th Grade':                   '1st-4th',
                '5th to 6th Grade':                   '5th-6th',
                '7th to 8th Grade':                   '7th-8th',
                '9th Grade':                          '9th',
                '10th Grade':                         '10th',
                '11th Grade':                         '11th',
                '12th Grade':                         '12th',
                'High School Graduate':               'HS-grad',
                'College (No Degree)':                'Some-college',
                'Associate (Vocational)':             'Assoc-voc',
                'Associate (Academic)':               'Assoc-acdm',
                'Bachelors':                          'Bachelors',
                'Masters':                            'Masters',
                'Professional Degree (MD, JD, etc.)': 'Prof-school',
                'Doctorate':                          'Doctorate',
            },
        },
    },
    # ---- ROW 3 ---------------------------------------------------------------
    {
        'workclass': {
            'name': 'Workclass',
            'type': 'dropdown',
            'nillable': True,
            'values': {
                'Government (Local)':   'Local-gov',
                'Government (State)':   'State-gov',
                'Government (Federal)': 'Federal-gov',
                'Private Sector':       'Private',
                'Self-Employed':        'Self-emp-not-inc',
                'Self-Employed (Inc)':  'Self-emp-inc',
                'Never Worked':         'Never-worked',
                'Without Pay':          'Without-pay',
            },
        },
        'occupation': {
            'name': 'Occupation',
            'type': 'dropdown',
            'nillable': True,
            'values': {
                'Accounting':                     'Prof-specialty',
                'Administrative':                 'Adm-clerical',
                'Analyst':                        'Prof-specialty',
                'Armed Forces':                   'Armed-Forces',
                'Clerical':                       'Adm-clerical',
                'Craft Repair':                   'Craft-repair',
                'Domestic Household Services':    'Priv-house-serv',
                'Engineering':                    'Prof-specialty',
                'Executive':                      'Exec-managerial',
                'Farming':                        'Farming-fishing',
                'Fishing':                        'Farming-fishing',
                'General Labor':                  'Handlers-cleaners',
                'Janitorial':                     'Handlers-cleaners',
                'Law Enforcement':                'Protective-serv',
                'Logistics':                      'Transport-moving',
                'Machine Operation / Inspection': 'Machine-op-inspct',
                'Managerial':                     'Exec-managerial',
                'Nurse':                          'Prof-specialty',
                'Other Service':                  'Other-service',
                'Public Safety':                  'Protective-serv',
                'Sales':                          'Sales',
                'Sciences':                       'Prof-specialty',
                'Security':                       'Protective-serv',
                'Software Development':           'Prof-specialty',
                'Technical Support':              'Tech-support',
                'Transportation':                 'Transport-moving',
            },
        },
    },
    # ---- ROW 4 ---------------------------------------------------------------
    {
        'hours-per-week': {
            'name': 'Hours per Week',
            'type': 'number',
            'min': 0,
            'max': 100,
        },
        'capital-gain': {
            'name': 'Capital Gain',
            'type': 'number',
            'min': 0,
            'max': 100000,
        },
        'capital-loss': {
            'name': 'Capital Loss',
            'type': 'number',
            'min': 0,
            'max': 50000,
        },
    }
]


logger = logging.getLogger(__package__)


class IncomePredictionUI(gr.Blocks):
    """Gradio UI for the Income Prediction Model"""

    def __init__(self, model: 'IncomePredictionModel', df: 'pd.DataFrame', /, **kwargs):
        """Initialize the UI

        Args:
            model: pre-trained income prediction model
            df:    dataset for evaluation
        """
        super().__init__(**kwargs)
        self._model = model
        self._df = df
        self._elems: dict[str, gr.Component] = {}
        with self:
            gr.Markdown("""\
## Income Prediction Model UIs

This UI allows you to interactively test the trained model on the test dataset.
The model predicts whether an individual earns more than $50K per year based on their features.
You can predict the income category for an individual by entering their features below:
""")
            # Inputs
            for row in INCOME_PREDICTION_UI_ELEMENTS:
                with gr.Row(equal_height=True):
                    for key, descriptor in row.items():
                        # create UI elements
                        match descriptor['type']:
                            case 'number':
                                default = random.randint(descriptor['min'], descriptor['max'])
                                kwargs = {
                                    'label': descriptor['name'], 'precision': 0, 'value': default,
                                    'minimum': descriptor['min'], 'maximum': descriptor['max'],
                                }
                                if (descriptor.get('randomize', False)):
                                    with gr.Group():
                                        self._elems[key] = gr.Number(**kwargs)
                                        btn_random = gr.Button('🎉 Randomize', size='sm')
                                        btn_random.click(partial(self._randomize, keys=(key,)),
                                                         outputs=self._elems[key],
                                                         show_progress='hidden')
                                else:
                                    self._elems[key] = gr.Number(**kwargs)
                            case 'dropdown':
                                values = descriptor['values']
                                default = random.choice(tuple(values.values()))
                                if (descriptor['nillable']):
                                    default = None
                                    values = {'': None, **values}
                                self._elems[key] = gr.Dropdown(label=descriptor['name'], value=default,
                                                               choices=tuple(values.items()))
            # Buttons
            with gr.Row(equal_height=True):
                btn_predict = gr.Button('🤖 Predict Yearly Salary', variant='primary')
                btn_randomize = gr.Button('🎉 Shuffle Inputs')
            # Output
            txt_result = gr.Label(label="Predicted Yearly Income Category")
            # Click Handler
            btn_predict.click(self._predict, tuple(self._elems.values()), txt_result)
            btn_randomize.click(partial(self._randomize, keys=tuple(self._elems.keys())),
                                outputs=tuple(self._elems.values()),
                                show_progress='hidden')

    def _predict(self, *args):
        """Predict Income Category (Single Record)

        Args:
            *args: values of the input features
                   in the same order as self._elems

        Returns:
            Predicted Income Category, either '>50K' or '<=50K'
        """
        record = dict(zip(self._elems, args))
        logger.debug(f'Predicting with values: {record}')
        df = pd.DataFrame([record])
        preds = self._model.predict(df)
        logger.info(f'Prediction: {preds}')
        return preds[0]

    def _randomize(self, keys: tuple[str]):
        """Randomize Inputs"""
        results = []
        for key in keys:
            # retrieve the UI element
            elem = self._elems[key]
            # match by type
            match (elem):
                case gr.Number():
                    # generate a random integer
                    results.append(random.randint(elem.minimum, elem.maximum))
                case gr.Dropdown():
                    # generate another random choice
                    results.append(random.choice(tuple(v for k, v in elem.choices if v is not None)))
        # apparently, when the target is a single component,
        # gradio expects just the single return value
        if (len(keys) == 1):
            return results[0]
        # return list of values otherwise
        return results
