import sys

sys.path.append("./")

import numpy as np
from torch.utils.data import IterableDataset
from datetime import datetime, time, timedelta
from submission.config import config
from pathlib import Path
import pandas as pd
import xarray as xr
import json


class ChallengeDataset(IterableDataset):
    def __init__(self, dataset_type: str, year: int, sites=None, eval=False, eval_year=2021, eval_day=1, eval_hours = 24):
        assert dataset_type in ("aerosols", "hrv", "nonhrv", "pv", "weather"), "ERROR LOADING DATA: dataset type provided is not correct [not of 'aerosols', 'hrv', 'nonhrv', 'pv' or 'weather']"
        assert year == 2020 or year == 2021, "ERROR LOADING DATA: year provided not correct [not 2020 or 2021]"
        assert eval_year == 2020 or eval_year == 2021, "eval year not 2020 or 2021"
        assert eval_day < 28, "eval day not < 28, lets not try that :)"

        self.dataset_type = dataset_type
        self.eval = eval

        # Load pv data by concatenating all data in this folder
        if not eval:
            months = [pd.read_parquet(f"/data/climatehack/official_dataset/pv/{year}/{i}.parquet").drop("generation_wh", axis=1) for i in range(1, 13)]
            pv = pd.concat(months)

            nonhrv = xr.open_mfdataset(f"/data/climatehack/official_dataset/{dataset_type}/{year}/*.zarr.zip", engine="zarr", chunks="auto")

            nwp = xr.open_mfdataset(f"/data/climatehack/official_dataset/weather/{year}/*.zarr.zip", engine="zarr", chunks="auto")

        else:
            def timeSlice(year, month, day, hours):
                t = datetime(year, month, day, 0, 0)
                return slice(str(t), str(t + timedelta(hours=hours)))

            months = [pd.read_parquet(f"/data/climatehack/official_dataset/pv/{year}/{month}.parquet").drop("generation_wh", axis=1)[timeSlice(year, month, eval_day, eval_hours)] for month in range(1, 13)]
            pv = pd.concat(months)

            def xr_open(path):
                return xr.open_dataset(path,
                                       engine="zarr",
                                       consolidated=True)

            nonhrv_data = [xr_open(f"/data/climatehack/official_dataset/nonhrv/{year}/{month}.zarr.zip").sel(time=timeSlice(year, month, eval_day, eval_hours)) for month in range(1, 13)]

            nonhrv = xr.concat(nonhrv_data, dim="time")

            nwp_data = [xr_open(f"/data/climatehack/official_dataset/weather/{year}/{month}.zarr.zip").sel(time=timeSlice(year, month, eval_day, eval_hours)) for month in range(1, 13)]
            nwp = xr.concat(nwp_data, dim="time")

        # pre-computed indices corresponding to each solar PV site stored in indices.json
        with open("indices.json") as f:
            site_locations = {
                    data_source: {
                        int(site): (int(location[0]), int(location[1]))
                        for site, location in locations.items()
                        } for data_source, locations in json.load(f).items()
                    }

        self.pv = pv
        self.nonhrv = nonhrv
        self.nwp = nwp
        self._site_locations = site_locations
        self._sites = sites if sites else list(site_locations[dataset_type].keys())

    def _get_image_times(self):
        if not self.eval:
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

                    current_time += timedelta(minutes=config.train.minutes_resolution)

                date += timedelta(days=1)
        else:
            for t in self.nwp["time"][::config.train.eval_resolution // 5]:
                yield datetime.fromtimestamp(t.item() / 1e9)

    def __iter__(self):
        pv = self.pv
        nwp = self.nwp

        rand_time_thresh, rand_site_thresh = config.train.random_time_threshold, config.train.random_site_threshold

        for time in self._get_image_times():
            if (not self.eval) and np.random.uniform(0, 1) > rand_time_thresh:
                continue

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

            nonhrv_data = self.nonhrv["data"].sel(time=first_hour).to_numpy()

            hrs_6 = slice(str(time - timedelta(hours=1)), str(time + timedelta(hours=4, minutes=55)))
            nwp_data = nwp.sel(time=hrs_6)
            nwp_data = xr.concat([nwp_data[k] for k in config.train.weather_keys], dim="time").values

            if not (nwp_data.shape == (6 * len(config.train.weather_keys), 305, 289)):
                # print(f"nwp pre site shape error: {nwp_data.shape}")
                continue

            for site in self._sites:
                if (not self.eval) and np.random.uniform(0, 1) > rand_site_thresh:
                    continue

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
                if not (site in self._site_locations["weather"]):
                    continue

                if not (site in self._site_locations["nonhrv"]):
                    continue

                x, y = self._site_locations[self.dataset_type][site]
                x_nwp, y_nwp = self._site_locations["weather"][site]

                nonhrv_features = nonhrv_data[:, y - 64: y + 64, x - 64: x + 64, config.data.channel]
                nwp_features = nwp_data[:, y_nwp - 64 : y_nwp + 64, x_nwp - 64 : x_nwp + 64]
                # hrv_features = hrv_data[:, y - 64: y + 64, x - 64: x + 64, :]
                # hrv_features = hrv_data.reshape((hrv_data.shape[0], hrv_data.shape[1], hrv_data.shape[2]*hrv_data.shape[3]))

                if np.isnan(nonhrv_features).any():
                    continue
                if np.isnan(nwp_features).any():
                    # print(f'WARNING: NaN in nwp_features for {time=}, {site=}')
                    continue

                if not (nonhrv_features.shape == (12, 128, 128)):
                    # print(f"shapemismatch got shape {nonhrv_features.shape}")
                    continue
                # if not (hrv_features.shape == (12, 128, 128, 3)):
                    # print('hrv shape mismatch')
                    # continue

                if not (nwp_features.shape == (6 * len(config.train.weather_keys), 128, 128)):
                    continue

                date_string = time.strftime("%y%m%d%H%M")
                date_int = int(date_string)

                yield date_int, site, site_features, site_targets, nonhrv_features, nwp_features


if __name__ == "__main__":
    eval_data = ChallengeDataset("nonhrv", 2020, eval=True)
    a = iter(eval_data)
