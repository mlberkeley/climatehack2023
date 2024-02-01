import os
import sys

sys.path.append('./')

import numpy as np
from torch.utils.data import IterableDataset
from datetime import datetime, time, timedelta
from pathlib import Path
import pandas as pd
#import modin.pandas as pd
import xarray as xr
from ocf_blosc2 import Blosc2
import json
import pickle
import h5py
from tqdm import tqdm
from submission.keys import WEATHER_RANGES


BAKE_START = datetime(2020, 1, 1)
BAKE_END = datetime(2021, 12, 31)

WEATHER_KEYS = ["alb_rad", "aswdifd_s", "aswdir_s", "cape_con", "clch", "clcl", "clcm", "clct", "h_snow", "omega_1000", "omega_700", "omega_850", "omega_950", "pmsl", "relhum_2m", "runoff_g", "runoff_s", "t_2m", "t_500", "t_850", "t_950", "t_g", "td_2m", "tot_prec", "u_10m", "u_50", "u_500", "u_850", "u_950", "v_10m", "v_50", "v_500", "v_850", "v_950", "vmax_10m", "w_snow", "ww", "z0"]

class BakerDataset(IterableDataset):
    def __init__(self):
        root_dir = Path("/data/climatehack/official_dataset")
        self.pv = pd.concat([
            pd.read_parquet(f"{root_dir}/pv/{y}/{i}.parquet").drop("generation_wh", axis=1)
            for y in (2020, 2021)
            for i in range(1, 13)
        ])

        self.nonhrv = xr.open_mfdataset(f"{root_dir}/nonhrv/*/*.zarr.zip", engine="zarr", chunks="auto")

        self.weather = xr.open_mfdataset(f"{root_dir}/weather/*/*.zarr.zip", engine="zarr", chunks="auto")

        # pre-computed indices corresponding to each solar PV site stored in indices.json
        with open("indices.json") as f:
            self.site_locations = {
                data_source: {
                    int(site): (int(location[0]), int(location[1]))
                    for site, location in locations.items()
                } for data_source, locations in json.load(f).items()
            }
        self.sites = list(self.site_locations['nonhrv'].keys())

    def _get_image_times(self):
        min_date = BAKE_START
        max_date = BAKE_END

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
        nwp = self.weather

        # for each date, iterate over sites
        # for each site, get pv features and targets
        # for each site, get weather features
        # for each site, get nonhrv features
        # for each valid entry, store mapping [idx], date, site, channel ok? (11 + 38)
        for time in self._get_image_times():
            first_hour = slice(str(time), str(time + timedelta(minutes=55)))

            pv_features = pv.xs(
                first_hour,
            )  # type: ignore
            pv_targets = pv.xs(
                slice(  # type: ignore
                    str(time + timedelta(hours=1)),
                    str(time + timedelta(hours=4, minutes=55)),
                ),
            )

            nonhrv_data = self.nonhrv["data"].sel(time=first_hour).to_numpy()

            hrs_6 = slice(str(time - timedelta(hours=1)), str(time + timedelta(hours=4, minutes=55)))
            nwp_data = nwp.sel(time=hrs_6)
            nwp_data = xr.concat([nwp_data[k] for k in WEATHER_KEYS], dim="channel").values

            for site in self.sites:

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

                nonhrv_flags = np.ones(11, dtype=np.bool_)
                weather_flags = np.ones(38, dtype=np.bool_)

                if not (nwp_data.shape == (len(WEATHER_KEYS), 6, 305, 289)):
                    weather_flags[:] = False

                # Get a 128x128 HRV crop centred on the site over the previous hour
                if nonhrv_flags.any() and not (site in self.site_locations["nonhrv"]):
                    nonhrv_flags[:] = False
                if weather_flags.any() and not (site in self.site_locations["weather"]):
                    weather_flags[:] = False

                x, y = self.site_locations['nonhrv'][site]
                x_nwp, y_nwp = self.site_locations["weather"][site]

                nonhrv_features = nonhrv_data[:, y - 64: y + 64, x - 64: x + 64, :]
                nwp_features = nwp_data[:, :, y_nwp - 64 : y_nwp + 64, x_nwp - 64 : x_nwp + 64]

                if nonhrv_flags.any() and not (nonhrv_features.shape == (12, 128, 128, 11)):
                    nonhrv_flags[:] = False

                if weather_flags.any() and not (nwp_features.shape == (len(WEATHER_KEYS), 6, 128, 128)):
                    weather_flags[:] = False

                if nonhrv_flags.any():
                    nonhrv_flags = ~np.isnan(nonhrv_features).any(axis=(0, 1, 2))
                if weather_flags.any():
                    weather_flags = ~np.isnan(nwp_features).any(axis=(1, 2, 3))

                yield int(time.timestamp()), site, nonhrv_flags, weather_flags


def main(
        overwrite=False,
        skip_bake_index=False,
        skip_pv=False,
        skip_nonhrv=False,
        skip_weather=False,
    ):

    # load pv, weather, nonhrv data
    print('Loading data')
    dataset = BakerDataset()
    if overwrite:
        h5file = h5py.File("data.h5", "w")
    else:
        h5file = h5py.File("data.h5", "a")

    # BAKE INDEX
    print('Creating/Loading bake index')
    if not skip_bake_index:
        # iterate over dates and cosntruct bake index
        start = datetime.now()
        bake_index = []

        for i, entry in enumerate(iter(dataset)):
            if i % 10000 == 0:
                print(i)
            bake_index.append(entry)

        bake_index_dt = np.dtype([
            ('time', np.int32),
            ('site', np.int32),
            ('nonhrv_flags', np.bool_, (11,)),
            ('weather_flags', np.bool_, (38,)),
        ], align=True)
        bake_index = np.array(bake_index, dtype=bake_index_dt)
        # np.save("bake_index.npy", bake_index)

        end = datetime.now()

        ds = h5file.create_dataset(
                'bake_index',
                shape=(len(bake_index),),
                dtype=bake_index_dt,
                chunks=(min(10000, len(bake_index)),),
        )
        ds[:] = bake_index

        print(f'Bake index created in: {end - start}')
    else:
        # bake_index = np.load("bake_index.npy")
        bake_index = h5file['bake_index'][:]

    # PV
    if not skip_pv:
        dataset.pv.to_pickle("pv.pkl")

    # NONHRV
    # based on bake index, save all nonhrv data to a single hdf5 file
    # get timestamps that we need
    # store a map for each mapping from timestamps to indices in the hdf5 files
    if not skip_nonhrv:
        times = set(bake_index['time'][bake_index['nonhrv_flags'].any(axis=1)])
        times = sorted(list(times))

        if 'nonhrv' in h5file:
            del h5file['nonhrv']
        ds = h5file.create_dataset(
                "nonhrv",
                shape=(11, len(times), 12, 293, 333),
                dtype=np.uint8,
                chunks=(1, 16, 12, 293, 333),
        )
        ds.attrs['times'] = np.array(times, dtype=np.uint32)

        print('Writing nonhrv data to hdf5 file')
        for i, timeint in enumerate(tqdm(times)):
            time = datetime.fromtimestamp(timeint)
            d = dataset.nonhrv['data'].sel(time=slice(time, time + timedelta(minutes=55))).to_numpy()
            # t, y, x, c -> c, t, y, x
            ds[:, i] = (d * 255).astype(np.uint8).transpose(3, 0, 1, 2)

    # WEATHER
    if not skip_weather:
        # based on bake index, save all weather data to a single hdf5 file

        times = set()
        for timeint, _, _, weather_flags in bake_index:
            if not weather_flags.any():
                continue
            for j in range(-1, 5):
                times.add(timeint + j * 3600)
        times = sorted(list(times))

        if 'weather' in h5file:
            del h5file['weather']
        ds = h5file.create_dataset(
                "weather",
                shape=(38, len(times), 305, 289),
                dtype=np.uint8,
                chunks=(1, 32, 305, 289),
        )
        ds.attrs['times'] = np.array(times, dtype=np.uint32)

        print('Writing weather data to hdf5 file')
        for i, timeint in enumerate(tqdm(times)):
            time = datetime.fromtimestamp(timeint)
            nwp_data = dataset.weather.sel(time=time)
            nwp_data = xr.concat([
                    (nwp_data[k] - WEATHER_RANGES[k][0]) / (WEATHER_RANGES[k][1] - WEATHER_RANGES[k][0])
                    for k in WEATHER_KEYS
                ], dim="channel").values
            # c, y, x -> c, y, x
            ds[:, i] = (nwp_data * 255).astype(np.uint8)
            # ds[i] = nwp_data.astype(np.float16).transpose(1, 2, 0)

    h5file.close()



if __name__ == "__main__":
    # TODO: add a dtype option
    # TODO  make this all configured with click or something
    # TODO  play around with normalizations (log for some data, linear, etc)
    main(
        overwrite=False,
        skip_bake_index=True,
        skip_pv=True,
        skip_nonhrv=True,
        skip_weather=True,
    )
