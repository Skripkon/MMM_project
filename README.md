# Overview
This challenge aims to predict plant species in a given location and time using various possible predictors: satellite images and time series, climatic time series, and other rasterized environmental data: land cover, human footprint, bioclimatic, and soil variables.

# Competition Description

This challenge is all about predicting plant species presence. Given GPS coordinates and various predictors, e.g., satellite images, climatic time series, land cover, human footprint, etc., a participant/team must predict a set of species that should grow there. To do so, we provide observation data comprising approximately 5 million Presence-Only (PO) occurrences and around 90 thousand Presence-Absence (PA) survey records. For more info about the data, please see the Data tab.


The training data comprises species observations and environmental data. Below, we provide at least some explanation of the data.

# ❗Data availability❗

1. For all Presence Absence (PA) observations, we provide the Bioclim, Landsat, and Sentinel-2 cubes. 2. For the Presence Only (PO) observations, we provide only the environmental values as tabular data.

## Observations data

## The species related training data comprises:

Presence-Absence (PA) surveys: including around 100 thousand surveys with roughly 5,000 species of the European flora. The presence-absence data (PA) is provided to compensate for the problem of false absences in PO data and calibrate models to avoid associated biases.
Presence-Only (PO) occurrences: combines around five million observations of around 10,000 species primarily originating from PlantNet App. This data constitutes the larger piece of the training data and covers all countries of our study area, but it has been sampled opportunistically (without a standardized sampling protocol), leading to various sampling biases. The local absence of a species from PO data does not necessarily mean it is truly absent. An observer might not have reported it because it was difficult to "see" it at this time of the year, to identify it as not a monitoring target, or just unattractive.
There are two CSV files with species occurrence data available on Seafile for training.

The PO metadata are available in GLC25_PO_metadata_train.csv.
The PA metadata are available in GLC25_PA_metadata_train.csv.
Environmental data

In addition to species data, we provide spatially referenced geographic and environmental data as supplementary input variables. More precisely, for each species observation location, we provide:

Satellite image patch: 4-band (R, G, B, NIR) 64x64 tiff files at 10m resolution.
Landsat time series: Up to 20 years of values for six satellite bands (R, G, B, NIR, SWIR1, and SWIR2).
Bioclim time series: Up to 20 years of values for 4 bands (mean, min. and max temp., and total precipitation).
Environmental rasters Various climatic, pedologic, land use, and human footprint variables at the European scale. We provide scalar values, time series, and original rasters from which you may extract local 2D images.
There are four separate folders with the data. A detailed description is provided below.

The Satellite image patches in ./SatelitePatches/.
The Satellite time series in ./SateliteTimeSeries-Landsat/.
The Bioclim time series in ./BioclimTimeSeries/.
The Environmental rasters in ./EnvironmentalRasters/.
Satellite image patches:

640mx640m R,G,B, NIR patches (four bands) centered at the observation geolocation and taken the same year. The patches are provided in the four band TIFF file accessible in the folder /SatelliteImages/.

Format: 64x64 TIFF files.
Resolution: 10 meters per pixel
Source: Sentinel2 remote sensing data pre-processed by the Ecodatacube platform
Access: Each TIFF file corresponds to a unique observation location (via "surveyId"). To load the patches for a selected observation, take the "surveyId" from any occurrence CSV and load it following this rule --> '…/CD/AB/XXXXABCD.jpeg'. For example, the image location for the surveyId 3018575 is "./75/85/3018575.tiff". For all "surveyId" with less than four digits, you can use a similar rule. For a "surveyId" 1 is "./1/1.tiff".
Landsat time series

Each observation is associated with the time series of the satellite median point values over each season since the winter of 1999 for six satellite bands (R, G, B, NIR, SWIR1, and SWIR2). This data carries a high-resolution local signature of the past 20 years' succession of seasonal vegetation changes, potential extreme natural events (such as fires), or land-use changes.

Format: . TimeSeries-Cubes - Extracted values are aggregated into 3d tensors with axes as BAND, QUARTER, and YEAR.
Resolution: The original satellite data has a resolution of 30m per pixel
Source: Landsat remote sensing data pre-processed by the Ecodatacube platform
Access: /SateliteTimeSeries-Landsat/cubes/
Bioclim time series

Four climatic variables computed monthly (mean, minimum and maximum temperature, and total precipitation) from January 2000 to December 2019, yielding 960 low-resolution rasters covering Europe.

Format: TimeSeries-Cubes - Extracted values are aggregated into 3d tensors with axes as RASTER-TYPE, YEAR, and MONTH.
Resolution: ~1 kilometer
Source: Chelsa
Access: /BioclimTimeSeries/cubes/
Environmental values

For each observation, we provide additional environmental data as scalar values that have already been extracted from the rasters. We provide CSV files, one for each band raster type, including Climate, Elevation, Human Footprint, LandCover, and SoilGrids.

Bioclimatic rasters: 19 low-resolution rasters covering Europe; commonly used in species distribution modeling. Provided in longitude/latitude coordinates (WGS84).

Format: GeoTIFF files with compression and CSV file with extracted values.
Resolution: 30 arcsec (~ 1 kilometer)
Source: CHELSA
Access: /EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010
Soil rasters: Nine pedologic low-resolution rasters covering Europe. Provided variables describe the soil properties from 5 to 15cm depth and are determinants of plant species distributions. Check the definition.txt file for information about the provided variables (e.g., pH, clay, organic carbon, and nitrogen contents, etc.).

Format: GeoTIFF files with compression and CSV file with extracted values.
Resolution: ~1 kilometer
Source: Soilgrids
Access: /EnvironmentalRasters/Soilgrids
Elevation: High-resolution raster covering Europe.

Format: GeoTIFF file with compression, Int16 numeric storage (13.2GB) and CSV file with extracted values.
Resolution: 1 arc second (~30 meter)
Source: ASTER Global Digital Elevation Model V3
Access: /EnvironmentalRasters/Elevation
Land Cover: A medium-resolution multi-band land cover raster covering Europe. Each band describes either the land cover class prediction or its confidence under various classifications. We recommend the use of IGBP (17 classes) or LCCS (43 classes) layers, often used in species distribution modeling.

Format: GeoTIFF file with compression and CSV file with extracted values.
Resolution: ~500m
Source: MODIS Terra+Aqua 500m
Access: /EnvironmentalRasters/LandCover/
Human footprint: 22 high-resolution OpenStreetMap rasters describing variables that reflect human footprint-

Format: GeoTIFF files with compression and CSV file with extracted values.
Resolution: 10-30m
Source: Ecodatacube platform
Access: The folder /EnvironmentalRasters/HumanFootprint/