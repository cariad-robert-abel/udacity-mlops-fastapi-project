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

## Training

This project uses [`dvc`](https://dvc.org/) to manage pipelines and version control data during
training.
[Weights & Biases](https://wandb.ai/) is used for experiment tracking and hosting datasets / models
in production.

## License

Original files Copyright 2012–2020 Udacity, Inc.
My additions to documentation and code are [MIT](https://spdx.org/licenses/MIT).
See [LICENSE-Udacity](LICENSE-Udacity) resp. [LICENSE](LICENSE).
