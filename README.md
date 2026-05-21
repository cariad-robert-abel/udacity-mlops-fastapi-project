---
title: Income Prediction
emoji: 💵
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 6.14.0
python_version: '3.13'
app_file: src/income-prediction/__main__.py
pinned: false
license: mit
short_description: Machine Learning Pipeline for Income Prediction
---
<!-- This is the Hugging Face Space Configuration -->
<!-- See https://huggingface.co/docs/hub/en/spaces-config-reference -->

# Deploying a Scalable Machine Learning Pipeline in Production for Income Prediction

This repository contains the project associated with "Deploying a Scalable Machine Learning Pipeline
in Production" Udacity course. It's a fork of Udacity's [Starter Kit](https://github.com/udacity/nd0821-c3-starter-code).

This GitHub.com project is located at [cariad-robert-abel/udacity-mlops-fastapi-project](https://github.com/cariad-robert-abel/udacity-mlops-fastapi-project).  
The associated Weights & Biases project is located at [wandb.ai/cariad-robert-abel-cariad-se/income-prediction](https://wandb.ai/cariad-robert-abel-cariad-se/income-prediction).

This application is deployed to my Hugging Face Space [cariad-robert-abel/udacity-income-prediction](https://huggingface.co/spaces/cariad-robert-abel/income-prediction), which is hosted at [cariad-robert-abel-income-prediction.hf.space/](https://cariad-robert-abel-income-prediction.hf.space/).

This project uses [`dvc`](https://dvc.org/) to manage pipelines and version control data during
training.
[Weights & Biases](https://wandb.ai/) is used for experiment tracking and hosting datasets / models
in production.

## Preprocessing

The original data was cleaned up using the [EDA](./EDA.ipynb) notebook.
The train-test split is done using the `prep` pipeline stage, which can be run using:

```bash
dvc exp run prep
```

Check the [preprocessing.yml](./cfg/preprocessing.yml) parameter file for more information.

## Training

Training can be run using the `train` pipeline stage:

```bash
dvc exp run train
```

Check the [training.yml](./cfg/training.yml) parameter file for more information.

## Production Model

The production model is [income-prediction-model:production](https://wandb.ai/cariad-robert-abel-cariad-se/income-prediction/artifacts/model/income-prediction-model/production).
Please find its detailed model card [here](./docs/model_card.md).

## Performance Metrics

Performance metrics can be computed live using the [census-income-split:reference](https://wandb.ai/cariad-robert-abel-cariad-se/income-prediction/artifacts/cleaned-data/census-income-split/reference) dataset from the Gradio web UI or via the command-line (see below).

Please find an example output file [here](./docs/slice_output.txt) for a slice over the `education` column of the `test.csv`
of the reference dataset.

## Main Executable

The main executable is used to implement the steps above as well as running a server that provides
access to a graphical user interface as well as a REST-ful API.

```bash
python ./src/income-prediction/__main__.py prep [-h] --config CONFIG
python ./src/income-prediction/__main__.py train [-h] --config CONFIG
python ./src/income-prediction/__main__.py metrics [-h] [--data DATA] [--model MODEL] [--slice SLICE] [--output OUTPUT]
python ./src/income-prediction/__main__.py serve [-h] [--host HOST] [--port PORT] [--data DATA] [--model MODEL]
```

The options should be self-explanatory, but do consult the built-in help (via `--help`/`-h`) if in doubt.

## Sanity Check

Make sure to run `sanitycheck.py` with proper `PYTHONPATH` set and point to `tests/test_app.py`.
For example on Windows:

```powershell
${env:PYTHONPATH}="./src/income-prediction"
"./tests/test_app.py" | python ./sanitycheck.py
```

## License

Original files Copyright 2012–2020 Udacity, Inc.
My additions to documentation and code are [MIT](https://spdx.org/licenses/MIT).
See [LICENSE-Udacity](LICENSE-Udacity) resp. [LICENSE](LICENSE).
