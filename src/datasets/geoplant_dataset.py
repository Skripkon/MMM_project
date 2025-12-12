import os
from typing import Optional, Literal
from pathlib import Path

import pandas as pd

import torch

import kagglehub

from src.datasets.base_dataset import BaseDataset

import torch


def torch_interp(x, xp, fp):
    x = x.to(dtype=xp.dtype)
    inds = torch.searchsorted(xp, x)
    inds = torch.clamp(inds, 1, len(xp) - 1)
    x0, x1 = xp[inds - 1], xp[inds]
    y0, y1 = fp[inds - 1], fp[inds]
    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (x - x0)


def quantile_normalize(band, low=0.02, high=0.98):
    if band is None:
        return band

    band = band.flatten()

    sorted_band = torch.sort(band).values
    quantiles = torch.quantile(sorted_band, torch.linspace(low, high, len(sorted_band)))
    normalized_band = torch_interp(band, sorted_band, quantiles).reshape(band.shape)

    min_val, max_val = torch.min(normalized_band), torch.max(normalized_band)

    # Prevent division by zero if min_val == max_val
    if max_val == min_val:
        return torch.zeros_like(normalized_band, dtype=torch.float32)  # Return an array of zeros
    # Perform normalization (min-max scaling)
    return ((normalized_band - min_val) / (max_val - min_val)).to(torch.float32)


def normalize(data):
    # Ensure data is float for mean/std calculation to avoid dtype errors
    data = data.to(torch.float32)
    return (data - data.mean()) / (data.std() + 1e-6)


class GeoPlantDataset(BaseDataset):
    def __init__(self,
                 local_path: Optional[str] = None,
                 split: Literal["train", "test", "val"] = "train",
                 section: Literal["P0", "PA"] = "PA",
                 limit: Optional[int] = None,
                 instance_transforms=None,
                 use_for_training_adaptive_k: bool = False):
        
        self.use_for_training_adaptive_k = use_for_training_adaptive_k

        is_val = False
        if split == "val":
            is_val = True
            split = "train"
        
        self.split = split
        self.section = section

        self.instance_transforms = instance_transforms
        
        if local_path is not None:
            self.index_path = Path(local_path)
        else:
            self.index_path = Path(kagglehub.competition_download('geoplant-at-paiss'))

        self.satellite_path, self.bioclimatic_path, self.landsat_path = None, None, None
        if section == "PA":
            self.satellite_path = self.index_path / Path("SatelitePatches") / Path(f"PA-{split}")
            self.bioclimatic_path = self.index_path / Path("BioclimTimeSeries/cubes") / Path(f"PA-{split}")
            self.landsat_path = self.index_path / Path("SateliteTimeSeries-Landsat/cubes") / Path(f"PA-{split}")

        self.environmental_path = self.index_path / Path("EnvironmentalVariables")

        def indexes_to_tensor(indexes_list, num_classes):
            indexes_list = [int(idx)-1 for idx in indexes_list]
            tensor = torch.zeros(num_classes, dtype=torch.float32)
            tensor[indexes_list] = 1.0
            return tensor

        self.df = pd.read_csv(self.index_path / Path(f"GLC25_{section}_metadata_{split}.csv"))
        
        num_classes = 11255

        # Group by survey_id (place speciesId in list)
        columns = self.df.columns.tolist()

        if split != "test":  # `speciesId` is the target column
            columns.remove("speciesId")

        columns.remove("surveyId")
        
        aggregations = {col: "first" for col in columns}

        if split != "test":
            aggregations["speciesId"] = lambda x: x.tolist()
        
        self.df = self.df.groupby("surveyId").agg(aggregations).reset_index()
        
        data_start, data_end = 0, len(self.df)
        if split == "train":
            mid = int(data_end * 0.99)
            if is_val:
                data_start = mid
            else:
                data_end = mid
        df = self.df.iloc[data_start:data_end].reset_index(drop=True)

        if split != "test":
            df['speciesId'] = df['speciesId'].apply(lambda x: indexes_to_tensor(x, num_classes))

        # FIXME: Add environmental features

        self.index = df.to_dict(orient='records')
        self.limit = limit if limit is not None else len(self.index)

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self.index[ind]
        survey_id = data_dict["surveyId"]

        target = None
        if "speciesId" in data_dict:
            target = data_dict["speciesId"]
    
        satellite = self.load_satellite_patch(survey_id)
        bioclimatic = self.load_bioclimatic_cube(survey_id)
        landsat = self.load_landsat_cube(survey_id)
        
        satellite = normalize(satellite.to(torch.float32)) if satellite is not None else None
        bioclimatic = normalize(bioclimatic.to(torch.float32)) if bioclimatic is not None else None
        landsat = normalize(landsat.to(torch.float32)) if landsat is not None else None
        
        instance_data = {
            "satellite": satellite,  # shape: (4, 64, 64) or None
            "bioclimatic": bioclimatic,  # shape: (4, 19, 12) or None
            "landsat": landsat,  # shape: (6, 4, 21) or None

            "table_data": torch.tensor([
                data_dict["lon"],
                data_dict["lat"],
                data_dict["year"],
                data_dict["geoUncertaintyInM"],
                data_dict["areaInM2"],
                # data_dict["region"],  # FIXME: categorical
                # data_dict["country"], # FIXME: categorical
            ], dtype=torch.float32),

            "target": target if not self.use_for_training_adaptive_k else torch.Tensor([torch.sum(target)]),
            "survey_id": survey_id,
        }
        instance_data = self.preprocess_data(instance_data)

        return instance_data
    
    def load_satellite_patch(self, survey_id: str):
        survey_str = str(survey_id)
        
        if self.satellite_path is None:
            return None
        
        path: str = self.satellite_path / Path(f"{survey_str[-2:]}/{survey_str[-4:-2]}/{survey_id}.tiff")
        # assert os.path.exists(path), f"File does not exist: {path}"
        if not os.path.exists(path): # FIXME: handle missing files
            return None
        
        return self.load_tiff_image(path)
    
    def load_bioclimatic_cube(self, survey_id: str):
        if self.satellite_path is None:
            return None
        
        path = self.bioclimatic_path / Path(f"GLC25-PA-{self.split}-bioclimatic_monthly_{survey_id}_cube.pt")
        # assert os.path.exists(path), f"File does not exist: {path}"
        if not os.path.exists(path):
            return None
        return self.load_tensor(path)
    
    def load_landsat_cube(self, survey_id: str):
        if self.satellite_path is None:
            return None
        
        path = self.landsat_path / Path(f"GLC25-PA-{self.split}-landsat-time-series_{survey_id}_cube.pt")
        # assert os.path.exists(path), f"File does not exist: {path}"
        if not os.path.exists(path):
            return None
        return self.load_tensor(path)

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return self.limit

    def preprocess_data(self, instance_data):
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Args:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element) (possibly transformed via
                instance transform).
        """
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                instance_data[transform_name] = self.instance_transforms[
                    transform_name
                ](instance_data[transform_name])
        return instance_data
