from functools import partial
import os
from pathlib import Path

import pandas as pd
import numpy as np
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

# TODO: pass model path as CLI argument
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

model = M2(num_classes=11255)
checkpoint = torch.load(os.path.join(LOCAL_MODEL_PATH, f"{MODEL_NAME}.pth"), map_location=torch.device(DEVICE), weights_only=False)
state_dict = checkpoint.get("state_dict", checkpoint)
model.load_state_dict(state_dict, strict=False)
model.eval().to(DEVICE)

# TODO: pass model path as CLI argument
adaptive_k_model = M2(use_for_training_adaptive_k=True)
checkpoint = torch.load("/Users/23048869/Desktop/untitled folder/MMM_project/saved/mmm/testing/model_best.pth", map_location=torch.device(DEVICE), weights_only=False)
state_dict = checkpoint.get("state_dict", checkpoint)
adaptive_k_model.load_state_dict(state_dict, strict=False)
adaptive_k_model.eval().to(DEVICE)

predictions = []
with torch.no_grad():
    for inputs in tqdm(test_dataloader): 
        for key in ['satellite', 'bioclimatic', 'landsat']:
            inputs[key] = inputs[key].to(DEVICE)
        outputs = model(**inputs)["logits"]

        adaptive_k_outputs = adaptive_k_model(**inputs)["logits"] # [batch_size, 1]
        ks = adaptive_k_outputs.squeeze(-1).tolist()
        for k, output in zip(ks, outputs):
            _, topk_indices = torch.topk(output, k=int(k) + 8, dim=-1)  # why add +8?
                                                                        # It boosts the performance...
                                                                        # Apparently, train data contains samples with not that much species,
                                                                        # whereas test data contains samples with more species.
                                                                        # Usage of the PO data must help in resolving the issue.
            predictions.append(topk_indices.tolist())
    
        # _, topk_indices = torch.topk(outputs, k=25, dim=-1)
        # predictions.extend(topk_indices.tolist())

lengths = [len(pred) for pred in predictions]
print("Median:", np.median(lengths))
print("Mean:", np.mean(lengths))
print("Max:", np.max(lengths))
print("Min:", np.min(lengths))


submission = pd.DataFrame({
    "surveyId": test_dataset.df.surveyId.values,
    "predictions": [
        " ".join(str(i + 1) for i in pred)
        for pred in predictions
    ]
})
submission.to_csv("submission.csv", index=False)
