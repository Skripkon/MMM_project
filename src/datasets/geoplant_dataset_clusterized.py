import os
from typing import Optional, Literal
from pathlib import Path
import pickle

import pandas as pd
import swifter

import torch

import kagglehub

from src.datasets.base_dataset import BaseDataset


def normalize(data):
    # Ensure data is float for mean/std calculation to avoid dtype errors
    data = data.to(torch.float32)
    return (data - data.mean()) / (data.std() + 1e-6)


NUM_CLASSES = 11255


def indexes_to_tensor(indexes_list):
    indexes_list = [int(idx)-1 for idx in indexes_list]
    tensor = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    tensor[indexes_list] = 1.0
    return tensor


class GeoPlantDataset(BaseDataset):
    def __init__(self,
                 local_path: Optional[str] = None,
                 split: Literal["train", "test", "val"] = "train",
                 part: Literal["PA", "P0", "both"] = "both",
                 limit: Optional[int] = None,
                 instance_transforms=None):
        is_val = False
        if split == "val":
            is_val = True
            split = "train"
        
        self.split = split

        self.instance_transforms = instance_transforms

        print("Initializing GeoPlantDataset...")
        
        if local_path is not None:
            self.index_path = Path(local_path)
        else:
            self.index_path = Path(kagglehub.competition_download('geoplant-at-paiss'))

        cache_path = Path("./cache")
        os.makedirs(cache_path, exist_ok=True)
        cache_file = cache_path / Path(f"geoplant_dataset_{split}_clusterized.csv")
        extra_cache_file = cache_path / Path(f"geoplant_dataset_{split}_clusterized_extra.csv")

        if not cache_file.exists() or not extra_cache_file.exists():
            print(f"Loading dataset index from {self.index_path}")

            self.df = pd.read_csv(self.index_path / Path(f"GLC25_PA_metadata_train.csv"))[["surveyId", "speciesId"]]

            columns = self.df.columns.tolist()
            columns.remove("surveyId")
            aggregations = {col: "first" for col in columns}
            aggregations["speciesId"] = lambda x: set(x.tolist())
            
            self.df = self.df.groupby("surveyId").agg(aggregations).reset_index()
            self.df["surveyId"] = self.df["surveyId"].apply(lambda x: {x})

            print("Loading additional clustered presence-absence data...")

            self.extra_df = pd.read_csv(self.index_path / Path(f"GLC25_PO_clustered_metadata.csv"))
            self.extra_df = self.extra_df[self.extra_df["close_to_PA"] == True]
            self.extra_df["speciesId"] = self.extra_df["speciesId"].apply(lambda x: eval(x))
            self.extra_df["surveyId"] = self.extra_df["surveyId"].apply(lambda x: eval(x))
            self.extra_df = self.extra_df[["surveyId", "speciesId"]]

            print("Concatenating environmental features...")

            self.concat_with_environmental(self.df, "PA")
            self.df.dropna()
            with open(cache_file, "wb") as f:
                pickle.dump(self.df, f)
            del self.df
            self.concat_with_environmental(self.extra_df, "PO")
            self.extra_df.dropna()
            with open(extra_cache_file, "wb") as f:
                pickle.dump(self.extra_df, f)
            del self.extra_df

        print(f"Loading dataset index from cache file {cache_file}")
        if part != "P0":
            with open(cache_file, "rb") as f:
                self.df = pickle.load(f)
        if part != "PA":
            with open(extra_cache_file, "rb") as f:
                self.extra_df = pickle.load(f)
            
        if part == "both":
            df = pd.concat([self.df, self.extra_df], ignore_index=True).sample(frac=1).reset_index(drop=True)
        elif part == "PA":
            df = self.df
        elif part == "P0":
            df = self.extra_df
        
        # split only extra part (so we can see impove on that one)
        data_start, data_end = 0, len(df)
        if split == "train":
            mid = int(data_end * 0.98)
            if is_val:
                data_start = mid
            else:
                data_end = mid
        df = df.iloc[data_start:data_end].reset_index(drop=True)

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
    
        instance_data = {
            "satellite": None,  # shape: (4, 64, 64) or None
            "bioclimatic": None,  # shape: (4, 19, 12) or None
            "landsat": None,  # shape: (6, 4, 21) or None

            "table_data": torch.tensor([
                # data_dict["lon"],
                # data_dict["lat"],
                # data_dict["year"],
                # data_dict["geoUncertaintyInM"],
                # data_dict["areaInM2"],
                # data_dict["region"],  # FIXME: categorical
                # data_dict["country"], # FIXME: categorical
            ], dtype=torch.float32),

            "climate_features": data_dict["bioclimatic_features"],  # shape: (19,)
            "elevation_features": data_dict["elevation_features"],  # shape: (1,)
            "human_footprint_features": data_dict["human_footprint_features"],  # shape: (22,)
            "land_cover_features": data_dict["land_cover_features"],  # shape: (13,)
            "soil_grids_features": data_dict["soil_grids_features"],  # shape: (9,)

            "target": indexes_to_tensor(target),
            "survey_id": survey_id,
        }

        instance_data = self.preprocess_data(instance_data)

        return instance_data
    
    def concat_with_environmental(self, df, section):

        environmental_path = self.index_path / Path("EnvironmentalValues")
        
        print("Loading environmental data from:", environmental_path)

        def merge_features(survey_ids, feature_dict):
            features_list = [feature_dict[sid] for sid in survey_ids if sid in feature_dict]
            if len(features_list) == 0:
                return None
            return torch.mean(torch.stack(features_list), dim=0)

        for feature in ["bioclimatic", "elevation", "human_footprint", "land_cover", "soil_grids"]:
            dir_map = {
                "bioclimatic": "ClimateAverage_1981-2010",
                "elevation": "Elevation",
                "human_footprint": "HumanFootprint",
                "land_cover": "LandCover",
                "soil_grids": "SoilGrids"
            }
            file_name_map = {
                "bioclimatic": "bioclimatic",
                "elevation": "elevation",
                "human_footprint": "human_footprint",
                "land_cover": "landcover",
                "soil_grids": "soilgrids"
            }
            data = pd.read_csv(environmental_path / Path(dir_map[feature]) / Path(f"GLC25-{section}-train-{file_name_map[feature]}.csv")).dropna()

            print("Normalizing environmental features...")

            # Normalize environmental columns
            for col in data.columns:
                if col != "surveyId":
                    data[col] = (data[col] - data[col].mean()) / (data[col].std() + 1e-6)

            print("Converting environmental features to tensors...")

            # make a tensors form columns except surveyId
            data[f"{feature}_features"] = data.apply(
                lambda row: torch.tensor(row.drop("surveyId").values, dtype=torch.float32), axis=1)
            
            print("Joining environmental features to main dataframe...")

            # join environmental features to main df
            data_dict = dict(zip(data["surveyId"], data[f"{feature}_features"]))
        
            df[f"{feature}_features"] = df["surveyId"].swifter.apply(lambda sids: merge_features(sids, data_dict))
            del data, data_dict

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
