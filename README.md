# CSE 599S: Advanced Machine Learning - Assignment 2

## Overview
[assignment_description.pdf](assignment_description.pdf)

Starting from the inference-only [LLaMA codebase](https://github.com/facebookresearch/llama/) and corpora from [The Pile](https://the-eye.eu/public/AI/pile/), implement and train from scratch an AI chatbot. Then, propose and conduct experiments and an ablation study on the trained model.

## Training
- [training_structure.ipynb](training/training_structure.ipynb) contains the from-scratch implementation for data processing and model training.

- [model_train.py](training/model_train.py) and [tokenizer_zeropad.py](training/tokenizer_zeropad.py) are modified deployments of LLaMA's inference architecture.

- [tokenizer.model](training/tokenizer.model) is a pretrained sentencepiece tokenizer provided by the LLaMA team upon request.

Primary contributor: [pw091](https://github.com/pw091)

## Inference
[inference/](inference) contains modified versions of the training implementation for sampling from a trained model.

Primary contributor: [Anderson-Lee-Git](https://github.com/Anderson-Lee-Git)

## Experimentation
- [epochrenzy/](epochfrenzy) examines the impacts of multiple pass training on inference.
  - Primary contributor: [mmontelaro](https://github.com/mmontelaro)

- [hyperparametersearch/](hyperparametersearch) analyzes the relationships among model arguments through their effects on inference.
  - Primary contributor: [Anderson-Lee-Git](https://github.com/Anderson-Lee-Git)

## Ablation Study
[ablation/](ablation) contains ad hoc modifications of training_structure.ipynb for an exploratory study of tokenizer sequencing schemes.
  - Primary contributor: [pw091](https://github.com/pw091)

## Written Report
[submitted_report.pdf](submitted_report.pdf) details the development process, model performance, experimentation findings, and implications resulting from the ablation study.
