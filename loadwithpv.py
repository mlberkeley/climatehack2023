from torch.utils.data import Dataset
import xarray as xr
from ocf_blosc2 import Blosc2
from functools import reduce
import numpy as np
import json
from pathlib import Path
import pandas as pd
from datetime import datetime, time, timedelta


months_num = 12
class NonHrvDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_path="/data/climatehack/official_dataset/nonhrv"):
        self.data_path = data_path
        #self.index_map = [] #list of true indexes so index_map[0] = 7 would mean when we get(0) we actually get 7
        self.index_map = np.load("./data_loader_saves/index_map_2021.npy")
        self.month_map = np.load("./data_loader_saves/month_map_2021.npy")
        #self.month_map = []
        self.month_lens = []

        def xr_open(path):
            return xr.open_dataset(
                    path,
                    engine="zarr",
                    consolidated=True,
                    )


        year = 2021
        self.data_2021 = [xr_open(f"/data/climatehack/official_dataset/nonhrv/{year}/{month}.zarr.zip") for month in range(1, months_num + 1)]

        #self.data_len = sum([x.shape[0] for x in self.data_2021])

        def verify(month, real_idx):
            to_check = self.data_2021[month]["time"][real_idx:real_idx + 12]
            return len(to_check) == 12 and reduce(lambda a, b: a and (b - a).item() == 300000000000 and b, to_check) 
        

        next_real_index = 0

        if len(self.index_map) == 0:      ##if we're not preloading
            for cur_month in range(0, months_num):

                print("on month ", cur_month)

                start_real_index = next_real_index

                month_idx = 0
                month_len = self.data_2021[month_idx]["time"].shape[0]
                while month_idx < month_len:
                    if verify(cur_month, month_idx):
                        self.index_map.append(sum(self.month_lens) + next_real_index)
                        next_real_index += 1
                    month_idx += 1

                self.month_lens.append(month_len)
                self.month_map.append((cur_month, start_real_index))

            self.data_len = next_real_index

            self.month_map = np.array(self.month_map)
            self.index_map = np.array(self.index_map)

            np.save("month_map_2021.npy", self.month_map)
            np.save("index_map_2021.npy", self.index_map)

        else:
            self.data_len = len(self.index_map)
        

        with open("indices.json") as f:
            site_locations = {
                data_source: {
                    int(site): (int(location[0]), int(location[1]))
                    for site, location in locations.items()
                } for data_source, locations in json.load(f).items()
            }

        year = 2021
        data_dir = Path(f"/data/climatehack/official_dataset/pv/{year}")
        pv = pd.concat(
            pd.read_parquet(parquet_file).drop("generation_wh", axis=1)
            for parquet_file in data_dir.glob('*.parquet')
        )

        self.pv = pv
        self._site_locations = site_locations
        #self._sites = sites if sites else list(site_locations[dataset_type].keys())
        dataset_type = "nonhrv"
        self._sites = list(site_locations[dataset_type].keys())


    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        pv = self.pv

        idx = self.index_map[idx]
        for tup in self.month_map:
            if idx >= tup[1]:
                nonhrv = self.data_2021[tup[0]]["data"][idx - tup[1] : idx - tup[1] + 12]

                time = pd.to_datetime(np.datetime_as_string(self.data_2021[tup[0]]["time"][idx - tup[1]]))

                first_hour = slice(str(time), str(time + timedelta(minutes=55)))
                pv_features = pv.xs(first_hour, drop_level=False)  # type: ignore

                pv_features = pv.xs(first_hour, drop_level=False)  # type: ignore
                pv_targets = pv.xs(
                    slice(  # type: ignore
                        str(time + timedelta(hours=1)),
                        str(time + timedelta(hours=4, minutes=55)),
                    ),
                    drop_level=False,
                )

                print(type(time))
                print(time)
                print(repr(time))
                time2 = pd.Timestamp(time, tz='UTC')
                site_ls = pv[pv.index.get_level_values('timestamp') == time2]
                print(site_ls)
                #site = site_ls[np.random.randint(0, len(site_ls))][1]
                site = site_ls.index[0][1]

                print(site)

                site_features = pv_features.xs(site, level=1).to_numpy().squeeze(-1)
                site_targets = pv_targets.xs(site, level=1).to_numpy().squeeze(-1)

                x, y = self._site_locations["nonhrv"][site]
                nonhrv_features = nonhrv[:, y - 64 : y + 64, x - 64 : x + 64]
                    
                return nonhrv_features, pv_features 

