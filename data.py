import pandas as pd
import xarray as xr

from pathlib import Path
from challenge_dataset import ChallengeDataset
from torch.utils.data import DataLoader


def load_pv():
    # Assuming data already downloaded... see TODO some other file for this

    # Load pv data by concatenating all data in this folder
    # Can modify as needed to load specific data

    data_dir = Path("/data/climatehack/official_dataset/pv/2020")
    pv = pd.concat(
        pd.read_parquet(parquet_file).drop("generation_wh", axis=1) for parquet_file in data_dir.glob('*.parquet')
    )

    return pv


def load_data(dataset_type: str = "hrv", year: int = 2021):
    # Once again, this is opening multiple datasets at once
    # hrv = xr.open_dataset("data/satellite-hrv/2020/7.zarr.zip", engine="zarr", chunks="auto")
    # opens a single dataset

    data = xr.open_mfdataset(f"/data/climatehack/official_dataset/{dataset_type}/{year}/*.zarr.zip", engine="zarr", chunks="auto")

    return data


def create_dataset(dataset_type: str, year: int, site_locations, batch_size: int, shuffle: bool = True):
    assert dataset_type in ("aerosols", "hrv", "nonhrv", "pv", "weather"), "ERROR LOADING DATA: dataset type provided is not correct [not of 'aerosols', 'hrv', 'nonhrv', 'pv' or 'weather']"
    assert year == 2020 or year == 2021, "ERROR LOADING DATA: year provided not correct [not 2020 or 2021]"

    pv, data = load_pv(), load_data(dataset_type, year)

    dataset = ChallengeDataset(pv, data, site_locations=site_locations)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=shuffle)

    return dataloader
