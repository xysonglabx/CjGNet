# CjGNet

This is the code for "Learning Long-Range Halogen Effects on Molecular Acidity across a 15-Million-Compound Chemical Space" paper.

## Directory Structure

```shell
/
├── configs/
├── data/
│   └── pka/
│       └── raw/
├── models/
├── output/
├── HII.py
├── LICENSE
├── README.md
├── acid_base_hii_utils.py
├── config.py
├── cross_validate.py
├── dataset.py
├── dataset_similarity.py
├── environment.yml
├── evaluate_utils.py
├── loss.py
├── model.py
├── optimized_config.py
├── optimized_train.py
├── train.py
└── utils.py
```

## Overview

### Supported Graph Neural Network Backbones

The model factory supports four graph neural network backbones:

- **GIN**
- **NTN**
- **GAT**
- **GCN**

## Installation

### Create conda environment

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate cjgnet
```

## Dataset

The dataset used in this study contains **12,532 molecular records** collected from the **iBonD ** database and the dataset reported in the QupKake study.

Data sources:

* **iBonD (Internet Bond-Energy Databank, pKa and BDE):** [iBonD Database](http://ibond.nankai.edu.cn?utm_source=chatgpt.com)
* **Abarbanel OD, Hutchison GR. *QupKake: Integrating Machine Learning and Quantum Chemistry for Micro-pKa Predictions.*** [DOI: 10.1021/acs.jctc.4c00328](https://doi.org/10.1021/acs.jctc.4c00328?utm_source=chatgpt.com)

## Train a Model

Basic model training:

```bash
python train.py \
  --cfg configs/example.yaml
```

Specify the graph neural network backbone:

```bash
python train.py \
  --cfg configs/example.yaml \
  --model-type NTN
```

Available model types are:

```text
NTN
GIN
GAT
GCN
```

## Bayesian Optimization

Bayesian hyperparameter optimization is implemented in `optimized_train.py`.

Enable Bayesian optimization in the configuration file:

```yaml
BO:
  ENABLE: True
```

Then run:

```bash
python optimized_train.py \
  --cfg configs/example.yaml
```


## Evaluation

After training, the best checkpoint selected according to validation performance is loaded automatically for final evaluation.

The evaluation module reports regression metrics including:

- R²;
- MSE;
- RMSE;
- MAE.
