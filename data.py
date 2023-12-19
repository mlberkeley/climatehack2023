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
    def __init__(self, dataset_type: str, year: int, sites=None, eval=False, eval_year=2021, eval_day=1, eval_hours = 24):
        assert dataset_type in ("aerosols", "hrv", "nonhrv", "pv", "weather"), "ERROR LOADING DATA: dataset type provided is not correct [not of 'aerosols', 'hrv', 'nonhrv', 'pv' or 'weather']"
        assert year == 2020 or year == 2021, "ERROR LOADING DATA: year provided not correct [not 2020 or 2021]"
        assert eval_year == 2020 or eval_year == 2021, "eval year not 2020 or 2021"
        assert eval_day < 28, "eval day not < 28, lets not try that :)"

        self.dataset_type = dataset_type

        # Load pv data by concatenating all data in this folder
        if not eval:
            months = [pd.read_parquet(f"/data/climatehack/official_dataset/pv/{year}/{i}.parquet").drop("generation_wh", axis=1) for i in range(1, 13)]
            pv = pd.concat(months)

            data = xr.open_mfdataset(f"/data/climatehack/official_dataset/{dataset_type}/{year}/*.zarr.zip", engine="zarr", chunks="auto")

        else:
            def timeSlice(year, month, day, hours):
                time = datetime(year, month, day, 0, 0)
                return slice(str(time), str(time + timedelta(hours=hours)))

            months = [pd.read_parquet(f"/data/climatehack/official_dataset/pv/{year}/{month}.parquet").drop("generation_wh", axis=1)[timeSlice(year, month, eval_day, eval_hours)] for month in range(1, 13)]
            pv = pd.concat(months)

            def xr_open(path):
                return xr.open_dataset( path,
                    engine="zarr",
                    consolidated=True,)

            xr_data = [xr_open(f"/data/climatehack/official_dataset/nonhrv/{year}/{month}.zarr.zip").sel(time=timeSlice(year, month, eval_day, eval_hours)) for month in range(1, 13)]

            data = xr.concat(xr_data, dim = "time")

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

            pv_features = pv.xs(
                    first_hour,
                    # drop_level=False
            )  # type: ignore
            pv_targets = pv.xs(
                slice(  # type: ignore
                    str(time + timedelta(hours=1)),
                    str(time + timedelta(hours=4, minutes=55)),
                ),
                # drop_level=False,
            )

            hrv_data = self.data["data"].sel(time=first_hour).to_numpy()
            for site in self._sites:
                # Get solar PV features and targets
                if not (site in pv_features.index.get_level_values('ss_id')):
                    continue
                if not (site in pv_targets.index.get_level_values('ss_id')):
                    continue

                site_features = pv_features.xs(site).to_numpy().squeeze(-1)
                site_targets = pv_targets.xs(site).to_numpy().squeeze(-1)

                if not (site_features.shape == (12,) and site_targets.shape == (48,)):
                    # print('WARNING: pv out of range')
                    continue

                # Get a 128x128 HRV crop centred on the site over the previous hour
                if not (site in self._site_locations[self.dataset_type]):
                    # print("site not avail")
                    continue

                x, y = self._site_locations[self.dataset_type][site]

                hrv_features = hrv_data[:, y - 64: y + 64, x - 64: x + 64, config.data.channel]
                # if (hrv_features != hrv_features).any():
                if np.isnan(hrv_features[:,0,0]).any() or np.isnan(hrv_features[:,-1,-1]).any():
                    print(f'WARNING: NaN in hrv_features for {time=}, {site=}')
                    continue

                if not (hrv_features.shape == (12, 128, 128)):
                    # print('hrv shape mismatch')
                    continue


                date_string = time.strftime("%Y%m%d%H%M%S")
                date_int = int(date_string)

                yield date_int, site, site_features, hrv_features, site_targets


# if __name__ == "__main__":
#     import cv2
#     import imageio
#     import matplotlib.pyplot as plt
#     import numpy as np

#     # load and visualize a singlular training example
#     dataset = ChallengeDataset("nonhrv", 2020)
#     for date, site, site_features, hrv_features, site_targets in dataset:
#         print(site_features.shape, hrv_features.shape, site_targets.shape)
#         print(hrv_features.min(), hrv_features.max())

#         plt.plot(np.arange(0, 1, 1/12), site_features, color="red")
#         plt.plot(np.arange(1, 5, 1/12), site_targets, color="blue")
#         Path("vis").mkdir(exist_ok=True)
#         plt.savefig("vis/power.jpg")

#         hrv_features = (hrv_features * 255).astype(np.uint8)
#         hrv_image = np.hstack(hrv_features)
#         cv2.imwrite("vis/hrv.jpg", hrv_image)

#         with imageio.get_writer("vis/vis.gif", mode="I") as writer:
#             for idx, frame in enumerate(hrv_features):
#                 writer.append_data(frame)

#         break

if __name__ == "__main__":
    data = ChallengeDataset("nonhrv", 2020)
    eval_data = ChallengeDataset("nonhrv", 2020, eval=True)
