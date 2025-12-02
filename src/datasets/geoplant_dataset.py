import os
from typing import Optional, Literal
from pathlib import Path

import pandas as pd

import torch

import kagglehub

from src.datasets.base_dataset import BaseDataset


class GeoPlantDataset(BaseDataset):
    def __init__(self,
                 split: Literal["train", "test"],
                 limit: Optional[int] = None,
                 instance_transforms=None):
        self.split = split

        self.instance_transforms = instance_transforms
        
        self.index_path = Path(kagglehub.competition_download('geoplant-at-paiss'))
        self.satellite_path = self.index_path / Path("SatelitePatches") / Path(f"PA-{split}")
        self.bioclimatic_path = self.index_path / Path("BioclimTimeSeries/cubes") / Path(f"PA-{split}")
        self.landsat_path = self.index_path / Path("SateliteTimeSeries-Landsat/cubes") / Path(f"PA-{split}")
        self.environmental_path = self.index_path / Path("EnvironmentalVariables")

        def indexes_to_tensor(indexes_list, max_index):
            indexes_list = [int(idx)-1 for idx in indexes_list]
            tensor = torch.zeros(max_index, dtype=torch.float32)
            tensor[indexes_list] = 1.0
            return tensor

        df = pd.read_csv(self.index_path / Path(f"GLC25_PA_metadata_{split}.csv"))
        # Group by survey_id (place speciesId in list)
        max_species_id = int(df['speciesId'].max().item())

        columns = df.columns.tolist()
        columns.remove("speciesId")
        columns.remove("surveyId")
        
        aggregations = {col: "first" for col in columns}
        aggregations["speciesId"] = lambda x: x.tolist()
        
        df = df.groupby("surveyId").agg(aggregations).reset_index()
        df['speciesId'] = df['speciesId'].apply(lambda x: indexes_to_tensor(x, max_species_id))

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
        target = data_dict["speciesId"]
        
        satellite = self.load_satellite_patch(survey_id)
        bioclimatic = self.load_bioclimatic_cube(survey_id)
        landsat = self.load_landsat_cube(survey_id)

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

            "target": target,
            "survey_id": survey_id,
        }

        instance_data = self.preprocess_data(instance_data)

        return instance_data
    
    def load_satellite_patch(self, survey_id: str):
        survey_str = str(survey_id)
        
        path: str = self.satellite_path / Path(f"{survey_str[-2:]}/{survey_str[-4:-2]}/{survey_id}.tiff")
        # assert os.path.exists(path), f"File does not exist: {path}"
        if not os.path.exists(path): # FIXME: handle missing files
            return None
        
        return self.load_tiff_image(path)
    
    def load_bioclimatic_cube(self, survey_id: str):
        path = self.bioclimatic_path / Path(f"GLC25-PA-{self.split}-landsat-bioclimatic_monthly_{survey_id}_cube.pt")
        # assert os.path.exists(path), f"File does not exist: {path}"
        if not os.path.exists(path):
            return None
        return self.load_tensor(path)
    
    def load_landsat_cube(self, survey_id: str):
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
