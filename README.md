# GeoPlant Species Distribution Modeling (SDM)

Solution to the [GeoPlant at PAISS Kaggle competition](https://www.kaggle.com/competitions/geoplant-at-paiss).

## Table of Contents

- [Task Overview](#task-overview)
- [Data](#data)
- [Solution Overview](#solution-overview)
- [Training](#training)
- [Inference](#inference)

## Task overview

The task is known as Species Distribution Modeling (https://en.wikipedia.org/wiki/Species_distribution_modelling).

- Input: longitude + attitude (point on Earth)
- Output: a list of species that present at that point (multi-label prediction)

## Data

The model uses a lot of data associated with a particular point on Earth:
- Satellite image patches
- Bioclimatic time series cubes
- Landsat time series cubes
- Environmental features (climate, elevation, land cover, soil, human footprint)

## Solution overview

1. preview [Dec 11]

See `work_done_by_dec_11.pdf`

2. full [Dec 22]

> TODO

## Training

Train a model using Hydra configuration:

```bash
python3 train.py
```

Configurations are in the `config/` directory. Specify a different config:

```bash
python3 train.py --config-name=resnet
```

> [!NOTE]
> Currenct configurations are tased on `A100` GPUs with `80GB` memory. Expected training time is around `1 hour`. This configuration gives solid score of `0.15223`

## Inference

Run inference on test data (hardcode a model in the file):

```bash
python3 inference.py
```

This generates a `submission.csv` file with predictions.
