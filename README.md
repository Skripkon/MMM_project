<img width="1237" height="688" alt="image" src="https://github.com/user-attachments/assets/85f62004-e294-4c80-93ba-6e8323a37a4e" /># GeoPlant Species Distribution Modeling (SDM)

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

For this moment we have tested 3 different but pritty similar approaches. All of them are based on the simple concept:
1. For each given modality we choose specific processor which makes embeddings of raw data.
2. This embeddings are concatenated and passed through the classification head.

To describe the model we would use the following notations:
```
- <modality 1> -|> <processor 1> |
- <modality 2> -|> <processor 2> | -> <embeddings mixup mechanism 1> |
- <modality 3> ->  <processor 3> |                                   |
- <modality 4> ->  <processor 4> | -> <embeddings mixup mechanism 2> | -> <classification head> -> <predictions>
- ...
```
Which means that first and second modalities are processed using cross-attention between their processors, and the third and fourth modalities are processed independently. Then first and second embeddings are mixed up using some mechanism (it could be concatenation, addition, cross-attention, etc.) and the same for third and fourth embeddings. Finally, all mixed embeddings are concatenated and passed through the classification head to get final predictions.

> [!NOTE]
> This is just an example of notation, the actual models could for example make cross-attention between all modalities.

With that here is the list of approaches we have tested:
- Resnet only model (using SE blocks) 
```
- satellite   -> resnet-3-se |
- bioclimatic -> resnet-3-se |
- landsat     -> resnet-3-se | -> concatination | -> mlp-head -> multi-class predictions
```

- Resnet (using SE blocks) + dual-path (this approach is currently sota on audio processing, so it should be usefull for processing time series with multiple features) model  
```
- satellite   -> resnet-3-se |
- bioclimatic -> dual-path   |
- landsat     -> dual-path   | -> concatination | -> mlp-head -> multi-class predictions
```

- Resnet (using SE blocks) + dual-path + cross-attention of embeddings model  
```
- satellite   -> resnet-3-se |
- bioclimatic -> dual-path   | -> cross-attention |
- satellite   -> resnet-3-se |
- landsat     -> dual-path   | -> cross-attention | -> concatination | -> mlp-head -> multi-class predictions
```

Because of our goal of making final solution step-by-step, the architecture difference isn't, so significant, but this definetly gives use direction of experiments.

2. full [Dec 22]

<img width="1237" height="688" alt="image" src="https://github.com/user-attachments/assets/d9bd9307-9c5e-4ac1-bb8e-c0d45c61fce6" />


The final architecture is expected to be as described below:

- RDP-CA  
```
- tabular data ->  clip-ecoder |

- satellite    -|> clip-ecoder |
- bioclimatic  -|> dual-path   |

- satellite    -|> clip-ecoder |
- landsat      -|> dual-path   |

- environment  ->  gpt-small   | concatination | -> mlp-head -> multi-class predictions
```

## Training

Train a model using Hydra configuration:

```bash
python3 train.py
```

Configurations are in the `config/` directory. Specify a different config:

```bash
python3 train.py --config-name=resnet_dual_path
```

> [!NOTE]
> Current configurations are tested on `A100` GPUs with `80GB` memory.
> Expected training time is around `1 hour`.
> This configuration gives solid score of `0.15223`

## Inference

Run inference on test data (hardcode a model in the file):

```bash
python3 inference.py
```

This generates a `submission.csv` file with predictions.
