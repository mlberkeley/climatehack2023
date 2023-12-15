from torch.utils.data import Dataset
import xarray as xr
from ocf_blosc2 import Blosc2
from functools import reduce
import numpy as np

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




            

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        idx = self.index_map[idx]
        for tup in self.month_map:
            if idx >= tup[1]:
                return self.data_2021[tup[0]]["data"][idx - tup[1] : idx - tup[1] + 12]
