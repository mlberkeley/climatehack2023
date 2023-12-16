import numpy as np
from torch.utils.data import IterableDataset
from datetime import datetime, time, timedelta
from submission.config import config
from pathlib import Path
import pandas as pd
import xarray as xr
from ocf_blosc2 import Blosc2
import json


class ChallengeDataset(IterableDataset):
    def __init__(self, dataset_type: str, year: int, sites=None):
        assert dataset_type in ("aerosols", "hrv", "nonhrv", "pv", "weather"), "ERROR LOADING DATA: dataset type provided is not correct [not of 'aerosols', 'hrv', 'nonhrv', 'pv' or 'weather']"
        assert year == 2020 or year == 2021, "ERROR LOADING DATA: year provided not correct [not 2020 or 2021]"

        self.dataset_type = dataset_type

        # Load pv data by concatenating all data in this folder
        # Can modify as needed to load specific data
        data_dir = Path(f"/data/climatehack/official_dataset/pv/{year}")

        months = [pd.read_parquet(f"/data/climatehack/official_dataset/pv/{year}/{i}.parquet").drop("generation_wh", axis=1) for i in range(1, 13)]
        pv = pd.concat(months)

        #pv = pd.concat(
            #pd.read_parquet(parquet_file).drop("generation_wh", axis=1)
            #for parquet_file in [data_dir.glob(f'{i}.parquet') for i in range(1, 13)]
        #)

        # opens a single dataset
        # hrv = xr.open_dataset("data/satellite-hrv/2020/7.zarr.zip", engine="zarr", chunks="auto")

        data = xr.open_mfdataset(f"/data/climatehack/official_dataset/{dataset_type}/{year}/*.zarr.zip", engine="zarr", chunks="auto")
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
        self.data = data
        self._site_locations = site_locations
        self._sites = sites if sites else list(site_locations[dataset_type].keys())

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

            hrv_data = self.data["data"].sel(time=first_hour).to_numpy()
            for site in self._sites:
                # Get solar PV features and targets
                if not (site in pv_features.index.get_level_values('ss_id')):
                    continue
                if not (site in pv_targets.index.get_level_values('ss_id')):
                    continue
                
                site_features = pv_features.xs(site, level=1).to_numpy().squeeze(-1)
                site_targets = pv_targets.xs(site, level=1).to_numpy().squeeze(-1)

                if not (site_features.shape == (12,) and site_targets.shape == (48,)):
                    # print('WARNING: pv out of range')
                    continue

                # Get a 128x128 HRV crop centred on the site over the previous hour
                if not (site in self._site_locations[self.dataset_type]): 
                    # print("site not avail")
                    continue

                x, y = self._site_locations[self.dataset_type][site]

                hrv_features = hrv_data[:, y - 64: y + 64, x - 64: x + 64, config.data.channel]
                if (hrv_features != hrv_features).any():
                    print(f'WARNING: NaN in hrv_features for {time=}, {site=}')
                    continue

                if not (hrv_features.shape == (12, 128, 128)):
                    # print('hrv shape mismatch')
                    continue


                date_string = time.strftime("%Y%m%d%H%M%S")
                date_int = int(date_string)

                yield date_int, site, site_features, hrv_features, site_targets


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2

    # load and visualize a singlular training example
    dataset = ChallengeDataset("hrv", 2020)
    for site_features, hrv_features, site_targets in dataset:
        print(site_features.shape, hrv_features.shape, site_targets.shape)
        print(hrv_features.min(), hrv_features.max())

        plt.plot(np.arange(0, 1, 1/12), site_features, color="red")
        plt.plot(np.arange(1, 5, 1/12), site_targets, color="blue")
        Path("vis").mkdir(exist_ok=True)
        plt.savefig("vis/power.jpg")

        hrv_image = (np.hstack(hrv_features) * 255).astype(np.uint8)
        cv2.imwrite("vis/hrv.jpg", hrv_image)

        break
