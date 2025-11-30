import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch


environmental_features_paths = [
    "ClimateAverage_1981-2010",
    "Elevation", 
    "HumanFootprint",
    "LandCover",
    "SoilGrids"
]


def get_satellite_path(survey_id: str, split_folder: str = 'PA-train') -> str:
    survey_str = str(survey_id)
    
    last_2 = survey_str[-2:]
    if len(survey_str) >= 4:
        third_fourth_from_end = survey_str[-4:-2]
    elif len(survey_str) >= 3:
        third_fourth_from_end = survey_str[-3:-2]

    path: str = f"data/SatelitePatches/{split_folder}/{last_2}/{third_fourth_from_end}/{survey_id}.tiff"
    assert os.path.exists(path), f"File does not exist: {path}"
    return path


def read_satellite_image(survey_id: str, split_folder: str = 'PA-train') -> torch.tensor:
    path = get_satellite_path(survey_id, split_folder)
    with rasterio.open(path) as src:
        image = src.read()
    return torch.tensor(image)


def get_bioclimatic_time_series_cube_path(survey_id: str, split_folder: str = 'PA-train') -> str:
    path = f"data/BioclimTimeSeries/cubes/{split_folder}/GLC25-PA-train-bioclimatic_monthly_{survey_id}_cube.pt"
    assert os.path.exists(path), f"File does not exist: {path}"
    return path


def get_bioclimatic_time_series_cube(survey_id: str, split_folder: str = 'PA-train') -> torch.tensor:
    path = get_bioclimatic_time_series_cube_path(survey_id, split_folder)
    with open(path, 'rb') as f:
        data = torch.load(f)
    return data

def get_satellite_time_series_landsat_cube_path(survey_id: str, split_folder: str = 'PA-train') -> str:
    path = f"data/SateliteTimeSeries-Landsat/cubes/{split_folder}/GLC25-PA-train-landsat-time-series_{survey_id}_cube.pt"
    assert os.path.exists(path), f"File does not exist: {path}"
    return path


def get_satellite_time_series_landsat_cube(survey_id: str, split_folder: str = 'PA-train') -> torch.tensor:
    path = get_satellite_time_series_landsat_cube_path(survey_id, split_folder)
    with open(path, 'rb') as f:
        data = torch.load(f)
    return data


def read_environmental_values(split_folder: str = 'PA-train') -> pd.DataFrame:
    base_dir = Path("data") / Path("EnvironmentalValues")
    dataframes = []
    for feature in environmental_features_paths:
        feature_dir = base_dir / Path(feature)
        filename = next(feature_dir.glob(f"*{split_folder}*.csv"))
        df = pd.read_csv(filename)
        # Only keep surveyId in the first dataframe; for the rest, drop it

        if dataframes:
            assert df['surveyId'].equals(dataframes[0]['surveyId']), (
                f"surveyId columns are not the same in {filename} and {dataframes[0].name}"
            )
            df = df.drop(columns=["surveyId"])
        dataframes.append(df)

    return pd.concat(dataframes, axis=1)


def get_environmental_values_tensor(survey_id: str, env_data: pd.DataFrame) -> torch.tensor:
    return torch.tensor(env_data.query("surveyId == @survey_id").values).squeeze()
