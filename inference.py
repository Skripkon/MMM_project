from functools import partial
import os
from typing import Any
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from src.datasets.geoplant_dataset import GeoPlantDataset
from src.datasets.collate import collate_fn
from src.models.resnet_dual_path import MultiModalFusionModel as M1  # model_weights/dual_path.pth
from src.models.cool_model import MultiModalFusionModel as M2        # model_weights/resnet.pth

ROOT_DIR = Path(__file__).parent
LOCAL_DATA_PATH = ROOT_DIR / Path("data")
LOCAL_MODEL_PATH = ROOT_DIR / Path("model_weights")
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

MODEL_NAME = "resnet"

test_dataset = GeoPlantDataset(
    local_path=LOCAL_DATA_PATH,
    split="test"
)

inference_collate_fn = partial(collate_fn, mixup_alpha=0.0, mixup_prob=0.0)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    collate_fn=inference_collate_fn
)

model = M2()
checkpoint = torch.load(os.path.join(LOCAL_MODEL_PATH, f"{MODEL_NAME}.pth"), map_location=torch.device(DEVICE), weights_only=False)
state_dict = checkpoint.get("state_dict", checkpoint)
model.load_state_dict(state_dict, strict=False)
model.eval().to(DEVICE)


predictions = []
with torch.no_grad():
    for inputs in tqdm(test_dataloader): 
        for key in ['satellite', 'bioclimatic', 'landsat']:
            inputs[key] = inputs[key].to(DEVICE)
        outputs = model(**inputs)["logits"]
        _, topk_indices = torch.topk(outputs, k=25, dim=-1)
        predictions.extend(topk_indices.tolist())


submission = pd.DataFrame({
    "surveyId": test_dataset.df.surveyId.values,
    "predictions": [
        " ".join(str(i) for i in pred)
        for pred in predictions
    ]
})
submission.to_csv("submission.csv", index=False)
