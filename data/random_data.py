from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pickle
import h5py


WEATHER_KEYS = ["alb_rad", "aswdifd_s", "aswdir_s", "cape_con", "clch", "clcl", "clcm", "clct", "h_snow", "omega_1000", "omega_700", "omega_850", "omega_950", "pmsl", "relhum_2m", "runoff_g", "runoff_s", "t_2m", "t_500", "t_850", "t_950", "t_g", "td_2m", "tot_prec", "u_10m", "u_50", "u_500", "u_850", "u_950", "v_10m", "v_50", "v_500", "v_850", "v_950", "vmax_10m", "w_snow", "ww", "z0"]
weather_keys = ["clch", "clcl", "clcm", "clct", "h_snow", "w_snow", "t_g", "t_2m", "tot_prec"]
weather_inds = sorted([WEATHER_KEYS.index(key) for key in weather_keys])


@dataclass
class ClimatehackDatasetConfig:
    start_date: datetime
    end_date: datetime
    root_dir: Path
    features: list[str] # TODO: make this an enum


class ClimatehackDataset(Dataset):

    def __init__(self, config: ClimatehackDatasetConfig):
        self.config = config

        self.bake_index = np.load("bake_index.npy")
        self.bake_index = self.bake_index[
            (self.bake_index['time'] < 1609459200) &
            self.bake_index['nonhrv_flags'].all(axis=1) &
            self.bake_index['weather_flags'].all(axis=1)
        ]

        self.pv = pd.concat([
                pd.read_parquet(f"{self.config.root_dir}/pv/{y}/{m}.parquet").drop("generation_wh", axis=1)
                for y in (2020, 2021)
                for m in range(1, 13)
        ])

        # self.data = h5py.File('/data/climatehack/nonhrv_weather.h5', 'r')
        self.data = h5py.File('data.uint8.2.h5', 'r')

        # 1609459200, 2986, 4642

        self.nonhrv_src = self.data['nonhrv']
        self.nonhrv_time_map = {t: i for i, t in enumerate(self.nonhrv_src.attrs['times'])}
        # todo only load filtered stuffs
        self.nonhrv = np.empty(
                (2, 2986, *self.nonhrv_src.shape[2:]),
                dtype=np.uint8
        )
        # self.nonhrv = self.nonhrv[[7,8], :2986]
        for i, a in enumerate((7, 8)):
            self.nonhrv_src.read_direct(self.nonhrv[i], np.s_[i, :2986, ...])
        # TODO convert that to a for loop as well lmao

        self.weather_src = self.data['weather']
        self.weather_time_map = {t: i for i, t in enumerate(self.weather_src.attrs['times'])}
        # self.weather = self.weather[weather_inds, :4642]
        self.weather = np.empty(
                (len(weather_inds), 4642, *self.weather_src.shape[2:]),
                dtype=np.uint8
        )
        for i, ind in enumerate(weather_inds):
            self.weather_src.read_direct(self.weather[i], np.s_[ind, :4642, ...])

        with open("indices.json") as f:
            self.site_locations = {
                data_source: {
                    int(site): (int(location[0]), int(location[1]))
                    for site, location in locations.items()
                } for data_source, locations in json.load(f).items()
            }

    def __len__(self):
        # TODO: filter dataset based on requirements
        return len(self.bake_index)

    def __getitem__(self, idx):
        timestamp, site, nonhrv_flags, weather_flags = self.bake_index[idx]
        time = datetime.fromtimestamp(timestamp)

        # pv
        first_hour = slice(
            str(time),
            str(time + timedelta(minutes=55))
        )
        pv_features = self.pv.xs(first_hour)
        pv_targets = self.pv.xs(
            slice(
                str(time + timedelta(hours=1)),
                str(time + timedelta(hours=4, minutes=55)),
            )
        )
        pv_features = pv_features.xs(site).to_numpy().squeeze(-1)
        pv_targets = pv_targets.xs(site).to_numpy().squeeze(-1)
        # pv_features = np.zeros((12,), dtype=np.float32)
        # pv_targets = np.zeros((48,), dtype=np.float32)

        # nonhrv
        x, y = self.site_locations['nonhrv'][site]
        nonhrv_ind = self.nonhrv_time_map[timestamp]
        nonhrv_features = self.nonhrv[
                1,
                nonhrv_ind,
                :,
                y - 64:y + 64,
                x - 64:x + 64,
                # 8
        ]
        # nonhrv.shape = (num_hours, hour, y, x, channels) = (*, 12, 293, 333, 11)
        # nonhrv_features.shape = (12, 128, 128)
        # nonhrv_features = np.zeros((12, 128, 128))

        # weather
        x, y = self.site_locations['weather'][site]
        weather_ind = self.weather_time_map[timestamp]
        weather_features = self.weather[
                :,
                weather_ind - 1:weather_ind + 5,
                y - 64:y + 64,
                x - 64:x + 64,
                # weather_inds
        ]
        # weather.shape = (num_hours, y, x, channels) = (*, 305, 289, 38)
        # weather_features.shape = (6, 128, 128, ?)
        weather_features = weather_features.transpose((0, 3, 1, 2))
        weather_features = weather_features.reshape((6 * len(weather_keys), 128, 128))

        return timestamp, site, pv_features, pv_targets, nonhrv_features, weather_features
