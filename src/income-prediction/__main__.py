#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import wandb
from sklearn.model_selection import train_test_split

from model import IncomePredictionModel
from __init__ import *  # noqa: F401,F403


logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format='[%(asctime)s][%(levelname)-8s] %(message)s', datefmt='%d %b %Y %H:%M:%S')
logger = logging.getLogger(__package__)


def prepare_data(config_path: Path) -> None:
    """Split and Store Prepared Data for Training and Testing

    Args:
        config_path: path to the preprocessing configuration file

    Raises:
        OSError: if configuration or output files cannot be accessed
        ValueError: if the configuration file is missing mandatory keys
    """
    logger.info(f'Preparing the training and test datasets from {config_path}...')

    # read configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # check configuration
    mandatory_keys = ('data', 'split', 'random-seed', 'test-size', 'stratify')
    missing_keys = tuple(k for k in mandatory_keys if k not in config)
    if missing_keys:
        raise ValueError(f'Missing mandatory keys in the configuration: {", ".join(missing_keys)}')

    # seed random number generator
    # https://scikit-learn.org/stable/glossary.html#term-random_state
    np.random.seed(config['random-seed'])

    # load dataset
    run = wandb.init(job_type='train-test-split')
    filename = run.use_artifact(config['data'], type='cleaned-data').file()
    data = pd.read_csv(filename)

    # store config
    run.config.update(config)

    train: pd.DataFrame
    test: pd.DataFrame
    stratify_df = data[config['stratify']] if config['stratify'] else None
    train, test = train_test_split(data, test_size=config['test-size'], stratify=stratify_df)

    # create metrics
    metrics = {
        'n_total': len(train) + len(test),
        'n_test': len(test),
        'n_train': len(train),
    }

    # add metrics to summary
    run.summary.update(metrics)

    # create output directory
    outdir = Path(__file__).parent / 'data'
    outdir.mkdir(parents=True, exist_ok=True)

    # store metrics
    with open(outdir / 'metrics.yml', 'w') as f:
        yaml.safe_dump(metrics, f)

    # store dataset artifact
    train.to_csv(outdir / 'train.csv', index=False)
    test.to_csv(outdir / 'test.csv', index=False)

    artifact = wandb.Artifact(config['split'], type='cleaned-data',
                              description='Split Train/Test Dataset')
    artifact.add_file(outdir / 'train.csv')
    artifact.add_file(outdir / 'test.csv')
    run.log_artifact(artifact)
    artifact.wait()

    # finish the run
    run.finish()


def train_model(config_path: Path) -> None:
    """Train and Store the Income Prediction Model from Prepared Data

    Args:
        config_path: path to the training configuration file

    Raises:
        OSError: if configuration, metrics output, or training data files cannot be accessed
        ValueError: if the configuration file is missing mandatory keys
    """
    logger.info(f'Training the model with configuration from {config_path}...')

    # read configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # check configuration
    mandatory_keys = ('data', 'model', 'random-seed', 'random-forest-classifier')
    missing_keys = tuple(k for k in mandatory_keys if k not in config)
    if missing_keys:
        raise ValueError(f'Missing mandatory keys in the configuration: {", ".join(missing_keys)}')

    # seed random number generator
    # https://scikit-learn.org/stable/glossary.html#term-random_state
    np.random.seed(config['random-seed'])

    # load training data
    run = wandb.init(job_type='training')
    dirname = run.use_artifact(config['data'], type='cleaned-data').download()
    train = pd.read_csv(Path(dirname) / 'train.csv')
    test = pd.read_csv(Path(dirname) / 'test.csv')

    # store config
    run.config.update(config)

    # actually train the model
    model = IncomePredictionModel.from_config(config['random-forest-classifier'])
    model.train(train)
    precision, recall, f_beta = model.validate(test)

    # create metrics
    metrics = {
        'precision': precision,
        'recall': recall,
        'f-beta': f_beta
    }

    # add metrics to summary
    run.summary.update(metrics)

    # create output directory
    outdir = Path(__file__).parent / 'model'
    outdir.mkdir(parents=True, exist_ok=True)

    # store metrics
    with open(outdir / 'metrics.yml', 'w') as f:
        yaml.safe_dump(metrics, f)

    # store model artifact
    model.store_model(outdir / 'model.joblib')
    artifact = wandb.Artifact(config['model'], type='model',
                              description='Trained Pipeline for Income Prediction')
    artifact.add_file(outdir / 'model.joblib')
    run.log_artifact(artifact)
    artifact.wait()

    # finish the run
    run.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description='Income Classification App')
    subparsers = parser.add_subparsers(title='command', dest='command', description='Sub-Command')
    parser.set_defaults(command='serve')

    serve_parser = subparsers.add_parser('serve', help='Serve the App (default)')  # noqa: F841

    prep_parser = subparsers.add_parser('prep', help='Prepare the Data')
    prep_parser.add_argument('--config', type=Path, required=True, help='Path to the preprocessing configuration file')

    train_parser = subparsers.add_parser('train', help='Train the Model')
    train_parser.add_argument('--config', type=Path, required=True, help='Path to the training configuration file')

    args = parser.parse_args()

    # abort with fatal error message in case WANDB_PROJECT is not set
    if 'WANDB_PROJECT' not in os.environ:
        logger.critical('WANDB_PROJECT environment variable must be set!')
        return 1

    match (args.command):
        case 'prep':
            prepare_data(args.config)
        case 'train':
            train_model(args.config)
        case 'serve':
            # TODO: implement web server + FastAPI app
            raise NotImplementedError('Serve command is not implemented yet')
            pass
        case _:
            raise NotImplementedError(f'Unknown command: {args.command}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
