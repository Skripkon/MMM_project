import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    batch = {}
    
    # instance_data = {
    #     "satellite": satellite,  # shape: (4, 64, 64) or None
    #     "bioclimatic": bioclimatic,  # shape: (4, 19, 12) or None
    #     "landsat": landsat,  # shape: (6, 4, 21) or None

    #     "table_data": torch.tensor([
    #         data_dict["lon"],
    #         data_dict["lat"],
    #         data_dict["year"],
    #         data_dict["geoUncertaintyInM"],
    #         data_dict["areaInM2"],
    #         # data_dict["region"],  # FIXME: categorical
    #         # data_dict["country"], # FIXME: categorical
    #     ], dtype=torch.float32),

    #     "target": target,
    #     "survey_id": survey_id,
    # }

    batch["satellite"] = torch.stack([
        item["satellite"] if item["satellite"] is not None
        else torch.zeros((4, 64, 64)) # FIXME: better handling of missing data
    for item in dataset_items]).to(torch.float32) / 255.0
    batch["bioclimatic"] = torch.stack([
        item["bioclimatic"] if item["bioclimatic"] is not None
        else torch.zeros((4, 19, 12)) # FIXME: better handling of missing data
    for item in dataset_items]).to(torch.float32)
    batch["landsat"] = torch.stack([
        item["landsat"] if item["landsat"] is not None
        else torch.zeros((6, 4, 21)) # FIXME: better handling of missing data
    for item in dataset_items]).to(torch.float32)
    batch["table_data"] = torch.stack([
        item["table_data"] if item["table_data"] is not None
        else torch.zeros((5,)) # FIXME: better handling of missing data
    for item in dataset_items]).to(torch.float32)

    # replace none with zeros
    batch["table_data"] = batch["table_data"].nan_to_num(0.0) # FIXME: better handling of missing data

    batch["target"] = torch.stack([item["target"] for item in dataset_items])

    batch["survey_id"] = [item["survey_id"] for item in dataset_items]

    return batch
