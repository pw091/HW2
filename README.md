# CSE 599S: Advanced Machine Learning - Homework 2

## Overview
[assignment_description.pdf](assignment_description.pdf)

Starting from the inference-only [LLaMA codebase](https://github.com/facebookresearch/llama/) and corpora from [The Pile](https://the-eye.eu/public/AI/pile/), implement and train from scratch an AI chatbot. Then, propose and conduct experiments and an ablation study on the trained model.

## Training
-[training_structure.ipynb](training/training_structure.ipynb) contains from-scratch implementation for data processing and model training.

-[model_train.py](training/model_train.py) and [tokenizer_zeropad.py](training/tokenizer_zeropad.py) are modified deployments of LLaMA's inference architecture.

-[tokenizer.model](training/tokenizer.model) is a pretrained sentencepiece tokenizer provided by the LLaMA team upon request.

## Inference
[inference/](inference) contains modified versions of the training implementation for sampling from a trained model.

## Experimentation
Analyses of [multiple pass training](epochfrenzy) and [hyperparameter tuning](hyperparametersearch).

-[epochrenzy/](epochfrenzy) examines the impacts of training redundancy on inference.

-[hyperparametersearch/](hyperparametersearch) analyzes the relationships among model arguments through their effects on inference.

## Ablation Study
[ablation/](ablation) contains ad hoc modifications of training_structure.ipynb for an exploratory study of tokenizer sequencing schemes.

## Written Report
[submitted_report.pdf](submitted_report.pdf) details the development process, model performance, experimentation findings, and implications resulting from the ablation study.
