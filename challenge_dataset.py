from torch.utils.data import IterableDataset
from datetime import datetime, time, timedelta
from config import config
from pathlib import Path
import pandas as pd
import xarray as xr
from ocf_blosc2 import Blosc2
import json


class ChallengeDataset(IterableDataset):
    def __init__(self, sites=None):
        # Assuming data already downloaded... see TODO some other file for this

        # Load pv data by concatenating all data in this folder
        # Can modify as needed to load specific data
        data_dir = Path(config.data.pv_path) / "2020"
        pv = pd.concat(
            pd.read_parquet(parquet_file).drop("generation_wh", axis=1)
            for parquet_file in data_dir.glob('*.parquet')
        )

        # Once again, this is opening multiple datasets at once
        # hrv = xr.open_dataset("data/satellite-hrv/2020/7.zarr.zip", engine="zarr", chunks="auto")
        # opens a single dataset
        hrv = xr.open_mfdataset(config.data.hrv_path + "/2020/*.zarr.zip", engine="zarr", chunks="auto")
        # nonhrv = xr.open_mfdataset("/data/climatehack/official_dataset/nonhrv/2020/*.zarr.zip", engine="zarr", chunks="auto")

        # pre-computed indices corresponding to each solar PV site stored in indices.json
        with open("indices.json") as f:
            site_locations = {
                data_source: {
                    int(site): (int(location[0]), int(location[1]))
                    for site, location in locations.items()
                } for data_source, locations in json.load(f).items()
            }
        self.pv = pv
        self.hrv = hrv
        # self.nonhrv = nonhrv
        self._site_locations = site_locations
        self._sites = sites if sites else list(site_locations["hrv"].keys())

    def _get_image_times(self):
        min_date = config.data.start_date
        max_date = config.data.end_date

        start_time = time(8)
        end_time = time(17)

        date = min_date
        while date <= max_date:
            current_time = datetime.combine(date, start_time)
            while current_time.time() < end_time:
                if current_time:
                    yield current_time

                current_time += timedelta(minutes=60)

            date += timedelta(days=1)

    def __iter__(self):
        pv = self.pv
        for time in self._get_image_times():
            first_hour = slice(str(time), str(time + timedelta(minutes=55)))

            pv_features = pv.xs(first_hour, drop_level=False)  # type: ignore
            pv_targets = pv.xs(
                slice(  # type: ignore
                    str(time + timedelta(hours=1)),
                    str(time + timedelta(hours=4, minutes=55)),
                ),
                drop_level=False,
            )

            hrv_data = self.hrv["data"].sel(time=first_hour).to_numpy()

            for site in self._sites:
                try:
                    # Get solar PV features and targets
                    site_features = pv_features.xs(site, level=1).to_numpy().squeeze(-1)
                    site_targets = pv_targets.xs(site, level=1).to_numpy().squeeze(-1)
                    assert site_features.shape == (12,) and site_targets.shape == (48,)

                    # Get a 128x128 HRV crop centred on the site over the previous hour
                    x, y = self._site_locations["hrv"][site]
                    hrv_features = hrv_data[:, y - 64: y + 64, x - 64: x + 64, 0]
                    assert hrv_features.shape == (12, 128, 128)

                    # How might you adapt this for the non-HRV, weather and aerosol data?
                except:
                    continue

                yield site_features, hrv_features, site_targets

