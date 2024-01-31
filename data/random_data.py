from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pickle
import h5py
from util import util
import submission.keys as keys


# TODO possibly get rid of this shit
# WEATHER_KEYS = ["alb_rad", "aswdifd_s", "aswdir_s", "cape_con", "clch", "clcl", "clcm", "clct", "h_snow", "omega_1000", "omega_700", "omega_850", "omega_950", "pmsl", "relhum_2m", "runoff_g", "runoff_s", "t_2m", "t_500", "t_850", "t_950", "t_g", "td_2m", "tot_prec", "u_10m", "u_50", "u_500", "u_850", "u_950", "v_10m", "v_50", "v_500", "v_850", "v_950", "vmax_10m", "w_snow", "ww", "z0"]
# weather_keys = ["clch", "clcl", "clcm", "clct", "h_snow", "w_snow", "t_g", "t_2m", "tot_prec"]
# weather_keys = ["clch", "clcl", "clcm", "clct"]
# weather_inds = sorted([WEATHER_KEYS.index(key) for key in weather_keys])


class ClimatehackDataset(Dataset):

    def __init__(self,
        start_date: datetime,
        end_date: datetime,
        root_dir: Path,
        meta_features: set[keys.META],
        nonhrv_features: set[keys.NONHRV],
        weather_features: set[keys.WEATHER],
        subset_size: int = 0,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.root_dir = root_dir
        self.meta_features = meta_features
        self.nonhrv_features = nonhrv_features
        self.weather_features = weather_features

        self.meta = pd.read_csv("/data/climatehack/official_dataset/pv/meta.csv")

        datafile = h5py.File(f'{root_dir}/baked_data.h5', 'r')

        start_time = datetime.now()
        self.bake_index = np.empty_like(datafile['bake_index'])
        datafile['bake_index'].read_direct(self.bake_index)
        self._filter_bake_index()
        if (subset_size > 0 and len(self.bake_index) > subset_size):
            rng = np.random.default_rng(21)
            rng.shuffle(self.bake_index)
            self.bake_index = self.bake_index[:subset_size]
        print(f"Loaded bake index in {datetime.now() - start_time}")

        # TODO replace with h5py... soon..
        start_time = datetime.now()
        self.pv = pd.read_pickle(f"{root_dir}/pv.pkl")
        print(f"Loaded pv in {datetime.now() - start_time}")

        start_time = datetime.now()
        nonhrv_src = datafile['nonhrv']
        self.nonhrv, self.nonhrv_time_map = self._load_data(           #output dim (len(channels), end_i - start_i, *src.shape[2:])
                nonhrv_src,
                [ch.value for ch in self.nonhrv_features],
                start_date,
                end_date,
        )
        print(f"Loaded nonhrv in {datetime.now() - start_time}")

        start_time = datetime.now()
        weather_src = datafile['weather']
        self.weather, self.weather_time_map = self._load_data(
                weather_src,
                [ch.value for ch in self.weather_features],
                start_date,
                end_date,
        )
        print(f"Loaded weather in {datetime.now() - start_time}")

        # TODO move this to data.h5
        with open("indices.json") as f:
            self.site_locations = {
                data_source: {
                    int(site): (int(location[0]), int(location[1]))
                    for site, location in locations.items()
                } for data_source, locations in json.load(f).items()
            }

        datafile.close()

    def _load_data(self, src: h5py.Dataset, channels: list[int], start_time: datetime, end_time: datetime):
        times = src.attrs['times']
        start_i = np.argmax(times >= start_time.timestamp())
        end_i = times.shape[0] - np.argmax(times[::-1] < end_time.timestamp())

        output = np.empty(
                (len(channels), end_i - start_i, *src.shape[2:]),
                dtype=np.uint8
        )
        # this does: out = src[selected_channels, selected_times] but faster
        # (unless selected channels are consequtive, but let's ignore that tiny detail)
        for i, ch in enumerate(channels):
            src.read_direct(output[i], np.s_[ch, start_i:end_i])

        time_map = {t: i for i, t in enumerate(times[start_i:end_i])}
        return output, time_map

    def _filter_bake_index(self):
        self.bake_index = self.bake_index[
            (self.bake_index['time'] >= self.start_date.timestamp()) &
            (self.bake_index['time'] < self.end_date.timestamp()) &
            self.bake_index['nonhrv_flags'].all(axis=1) &
            self.bake_index['weather_flags'].all(axis=1)
        ]

    def __len__(self):
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

        # nonhrv
        x, y = self.site_locations['nonhrv'][site]
        nonhrv_ind = self.nonhrv_time_map[timestamp]
        nonhrv_out_raw = self.nonhrv[
                :,
                nonhrv_ind,
                :,
                y - 64:y + 64,
                x - 64:x + 64,
        ]
        # nonhrv.shape = (channels, num_hours, hour, y, x) = (11, *, 12, 293, 333)
        # nonhrv_out_raw.shape = (12, 128, 128)
        nonhrv_out_raw = nonhrv_out_raw.astype(np.float32) / 255
        nonhrv_out = {
            key: nonhrv_out_raw[i]
            for i, key in enumerate(self.nonhrv_features)
        }

        # weather
        x, y = self.site_locations['weather'][site]
        weather_ind = self.weather_time_map[timestamp]
        weather_out_raw = self.weather[
                # [ch.value for ch in self.weather_features], # XXX
                :,
                weather_ind - 1:weather_ind + 5,
                y - 64:y + 64,
                x - 64:x + 64,
        ]
        # weather.shape = (channels, num_hours, y, x) = (38, *, 305, 289)
        # weather_out_raw.shape = (len(weather_keys), 6, 128, 128)
        # weather_out_raw = weather_out_raw.transpose((0, 3, 1, 2))
        # weather_out_raw = weather_out_raw.reshape((6 * len(self.weather_features), 128, 128))
        weather_out_raw = weather_out_raw.astype(np.float32) / 255
        weather_out = {
            key: weather_out_raw[i]
            for i, key in enumerate(self.weather_features)
        }

        # meta
        df = self.meta
        ss_id, lati, longi, _, orientation, tilt, kwp, _ = df.iloc[np.searchsorted(df['ss_id'].values, site)]
        # DO NOT CHANGE THE ORDER OF THESE
        site_features = [timestamp, lati, longi, orientation, tilt, kwp, ss_id]
        meta_out = {
            key: site_features[i]
            for i, key in enumerate(self.meta_features)
        }


        # return timestamp, site, pv_features, pv_targets, nonhrv_features, weather_features, site_features
        return pv_features, meta_out, nonhrv_out, weather_out, pv_targets
