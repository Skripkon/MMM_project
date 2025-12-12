import torch


def collate_fn(dataset_items: list[dict], mixup_alpha: float = 1.0, mixup_prob: float = 0.5) -> dict[str, torch.Tensor]:
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
        else torch.full((4, 64, 64), float('nan'))
    for item in dataset_items]).to(torch.float32) / 255.0
    batch["bioclimatic"] = torch.stack([
        item["bioclimatic"] if item["bioclimatic"] is not None
        else torch.full((4, 19, 12), float('nan'))
    for item in dataset_items]).to(torch.float32)
    batch["landsat"] = torch.stack([
        item["landsat"] if item["landsat"] is not None
        else torch.full((6, 4, 21), float('nan'))
    for item in dataset_items]).to(torch.float32)
    batch["table_data"] = torch.stack([
        item["table_data"] if item["table_data"] is not None
        else torch.full((5,), float('nan'))
    for item in dataset_items]).to(torch.float32)

    # replace none with zeros
    for name in ["satellite", "bioclimatic", "landsat", "table_data"]:
        batch[name] = batch[name].nan_to_num(0.0) # FIXME: better handling of missing data
        batch[name][(batch[name] == float('-inf')) | (batch[name] == float('inf'))] = 0.0 # FIXME: better handling of missing data

    if dataset_items[0]["target"] is not None:
        batch["target"] = torch.stack([item["target"] for item in dataset_items])

    batch["survey_id"] = [item["survey_id"] for item in dataset_items]

    # Apply mixup augmentation
    if mixup_alpha > 0.0 and torch.rand(1).item() < mixup_prob:
        lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()

        batch_size = batch["satellite"].size(0)
        index = torch.randperm(batch_size)

        for key in ["satellite", "bioclimatic", "landsat", "table_data", "target"]:
            if key != "target" or dataset_items[0]["target"] is not None:
                batch[key] = lam * batch[key] + (1 - lam) * batch[key][index]

    return batch
