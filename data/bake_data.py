import sys

sys.path.append('./')

from datetime import datetime, time, timedelta
import numpy as np
import torch
from torch.utils.data import IterableDataset
import pandas as pd
from pathlib import Path
import xarray as xr
from ocf_blosc2 import Blosc2
import json
import pickle
import h5py
from tqdm import tqdm
from submission.models.keys import META, WEATHER, WEATHER_RANGES, AEROSOLS, AEROSOLS_RANGES
from loguru import logger
from submission.modules.solar import solar_pos
import multiprocessing as mp
from easydict import EasyDict as edict
from types import SimpleNamespace


THREAD_COUNT = 8

BAKE_START = datetime(2020, 1, 1)
BAKE_END = datetime(2021, 12, 31)

def _get_image_times():
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


class Baker(object):

    def __init__(self):
        logger.debug(f'Initializing worker {mp.current_process().name}')

        root_dir = Path("/data/climatehack/official_dataset")
        self.pv = pd.concat([
            pd.read_parquet(f"{root_dir}/pv/{y}/{i}.parquet").drop("generation_wh", axis=1)
            for y in (2020, 2021)
            for i in range(1, 13)
        ])

        self.hrv = xr.open_mfdataset(f"{root_dir}/hrv/*/*.zarr.zip", engine="zarr", chunks="auto")
        self.nonhrv = xr.open_mfdataset(f"{root_dir}/nonhrv/*/*.zarr.zip", engine="zarr", chunks="auto")
        self.weather = xr.open_mfdataset(f"{root_dir}/weather/*/*.zarr.zip", engine="zarr", chunks="auto")
        self.aerosols = xr.open_mfdataset(f"{root_dir}/aerosols/*/*.zarr.zip", engine="zarr", chunks="auto")

        # pre-computed indices corresponding to each solar PV site stored in indices.json
        with open("indices.json") as f:
            self.site_locations = {
                data_source: {
                    int(site): (int(location[0]), int(location[1]))
                    for site, location in locations.items()
                } for data_source, locations in json.load(f).items()
            }
        self.sites = list(self.site_locations['nonhrv'].keys())
        self.meta = pd.read_csv(f"{root_dir}/pv/meta.csv")
        logger.debug(f'Done initializing worker {mp.current_process().name}')


    def __call__(self, time):
        out = []

        first_hour = slice(str(time), str(time + timedelta(minutes=55)))
        next_5_hours = slice(str(time + timedelta(hours=1)), str(time + timedelta(hours=4, minutes=55)))
        hrs_6 = slice(str(time - timedelta(hours=1)), str(time + timedelta(hours=4, minutes=55)))

        pv_features = self.pv.xs(first_hour)
        pv_targets = self.pv.xs(next_5_hours)

        hrv_data = self.hrv['data'].sel(time=first_hour).to_numpy()
        nonhrv_data = self.nonhrv["data"].sel(time=first_hour).to_numpy()
        weather_data = self.weather.sel(time=hrs_6)
        weather_data = xr.concat([weather_data[k.name.lower()] for k in WEATHER], dim="channel").values
        aerosols_data = self.aerosols.sel(time=hrs_6)
        aerosols_data = xr.concat([aerosols_data[k.name.lower()] for k in AEROSOLS], dim="channel").values

        # fast way to check for nans
        hrv_any_nan = np.isnan(np.dot(hrv_data.reshape(-1), hrv_data.reshape(-1)))
        nonhrv_any_nan = np.isnan(np.dot(nonhrv_data.reshape(-1), nonhrv_data.reshape(-1)))
        weather_any_nan = np.isnan(np.dot(weather_data.reshape(-1), weather_data.reshape(-1)))
        aerosols_any_nan = np.isnan(np.dot(aerosols_data.reshape(-1), aerosols_data.reshape(-1)))

        for i, site in enumerate(self.sites):

            # Get solar PV features and targets
            if not (site in pv_features.index.get_level_values('ss_id')):
                continue
            if not (site in pv_targets.index.get_level_values('ss_id')):
                continue
            site_features = pv_features.xs(site).to_numpy().squeeze(-1)
            site_targets = pv_targets.xs(site).to_numpy().squeeze(-1)
            if not (site_features.shape == (12,) and site_targets.shape == (48,)):
                continue

            hrv_flags = np.ones(1, dtype=np.bool_)
            nonhrv_flags = np.ones(11, dtype=np.bool_)
            weather_flags = np.ones(38, dtype=np.bool_)
            aerosols_flags = np.ones(13, dtype=np.bool_)

            # shape checks
            if not (hrv_data.shape == (12, 592, 684, 1)):
                hrv_flags[:] = False
            if not (nonhrv_data.shape == (12, 293, 333, 11)):
                nonhrv_flags[:] = False
            if not (weather_data.shape == (len(WEATHER), 6, 305, 289)):
                weather_flags[:] = False
            if not (aerosols_data.shape == (len(AEROSOLS), 6, 8, 250, 240)):
                aerosols_flags[:] = False

            # get crops
            if site in self.site_locations["hrv"]:
                x_hrv, y_hrv = self.site_locations["hrv"][site]
                hrv_features = hrv_data[:, y_hrv - 64: y_hrv + 64, x_hrv - 64: x_hrv + 64, :]
            else:
                hrv_flags[:] = False

            if site in self.site_locations["nonhrv"]:
                x_nonhrv, y_nonhrv = self.site_locations['nonhrv'][site]
                nonhrv_features = nonhrv_data[:, y_nonhrv - 64: y_nonhrv + 64, x_nonhrv - 64: x_nonhrv + 64, :]
            else:
                nonhrv_flags[:] = False

            if site in self.site_locations["weather"]:
                x_weather, y_weather = self.site_locations["weather"][site]
                weather_features = weather_data[:, :, y_weather - 64 : y_weather + 64, x_weather - 64 : x_weather + 64]
            else:
                weather_flags[:] = False

            if site in self.site_locations["aerosols"]:
                x_aerosols, y_aerosols = self.site_locations["aerosols"][site]
                aerosols_features = aerosols_data[:, :, :, y_aerosols - 64 : y_aerosols + 64, x_aerosols - 64 : x_aerosols + 64]
            else:
                aerosols_flags[:] = False

            # shape checks
            if hrv_features.shape != (12, 128, 128, 1):
                hrv_flags[:] = False
            if nonhrv_features.shape != (12, 128, 128, 11):
                nonhrv_flags[:] = False
            if weather_features.shape != (len(WEATHER), 6, 128, 128):
                weather_flags[:] = False
            if aerosols_features.shape != (len(AEROSOLS), 6, 8, 128, 128):
                aerosols_flags[:] = False

            # nan checks
            if hrv_flags.any() and hrv_any_nan:
                hrv_flags = ~np.isnan(hrv_features).any(axis=(0, 1, 2))
            if nonhrv_flags.any() and nonhrv_any_nan:
                nonhrv_flags = ~np.isnan(nonhrv_features).any(axis=(0, 1, 2))
            if weather_flags.any() and weather_any_nan:
                weather_flags = ~np.isnan(weather_features).any(axis=(1, 2, 3))
            if aerosols_flags.any() and aerosols_any_nan:
                aerosols_flags = ~np.isnan(aerosols_features).any(axis=(1, 2, 3, 4))

            if not (hrv_flags.any() or nonhrv_flags.any() or weather_flags.any() or aerosols_flags.any()):
                continue

            # get solar angles
            df = self.meta
            ss_id, lati, longi, _, orientation, tilt, kwp, _ = df.iloc[np.searchsorted(df['ss_id'].values, site)]
            # DO NOT CHANGE THE ORDER OF THESE
            site_features = [time.timestamp(), lati, longi, orientation, tilt, kwp, ss_id]
            meta = {
                key: torch.Tensor([site_features[i]])
                for i, key in enumerate(META)
            }
            solar_angles = solar_pos(meta, 'cpu', hourly=True).reshape((6, 2))

            out.append((int(time.timestamp()), site, hrv_flags, nonhrv_flags, weather_flags, aerosols_flags, solar_angles))

        return out


def get_bake_index(baker):

    # for time in _get_image_times():
    # with mp.Pool(16) as pool:
    #     for i, entry in enumerate(tqdm(pool.imap_unordered(bake_time, times, chunksize=1), total=len(times))):
    #         bake_index[i] = entry
    bake_index = []
    # with mp.Pool(1) as pool:
    #     for entry in tqdm(pool.map(Baker(), times, chunksize=1), total=len(times)):
    #         bake_index.extend(entry)
    # for entry in tqdm(map(Baker(), times), total=len(times)):
    #     bake_index.extend(entry)
    times = list(_get_image_times())
    with mp.pool.ThreadPool(THREAD_COUNT) as pool:
        for entry in tqdm(pool.imap_unordered(baker, times, chunksize=1), total=len(times)):
            bake_index.extend(entry)

    return bake_index


def main(
        overwrite=False,
        skip_bake_index=False,
        skip_pv=False,
        skip_hrv=False,
        skip_nonhrv=False,
        skip_weather=False,
        skip_aerosols=False,
    ):

    # load pv, weather, nonhrv data
    logger.info('Loading data')
    # dataset = BakerDataset()
    dataset = Baker()
    if overwrite:
        h5file = h5py.File("data.h5", "w")
    else:
        h5file = h5py.File("data.h5", "a")

    # BAKE INDEX
    logger.info('Creating/Loading bake index')
    if not skip_bake_index:
        # iterate over dates and cosntruct bake index
        start = datetime.now()
        # bake_index = []

        # for entry in tqdm(dataset):
        #     bake_index.append(entry)
        bake_index = get_bake_index(dataset)

        bake_index_dt = np.dtype([
            ('time', np.int32),
            ('site', np.int32),
            ('hrv_flags', np.bool_, (1,)),
            ('nonhrv_flags', np.bool_, (11,)),
            ('weather_flags', np.bool_, (38,)),
            ('aerosols_flags', np.bool_, (13,)),
            ('solar_angles', np.float32, (6, 2)),
        ], align=True)
        bake_index = np.array(bake_index, dtype=bake_index_dt)
        bake_index = np.sort(bake_index, order='time')
        np.save("bake_index.bac.npy", bake_index)

        if 'bake_index' in h5file:
            del h5file['bake_index']
        ds = h5file.create_dataset(
                'bake_index',
                shape=(len(bake_index),),
                dtype=bake_index_dt,
                chunks=(min(10000, len(bake_index)),),
        )
        ds[:] = bake_index

        end = datetime.now()
        logger.info(f'Bake index created in: {end - start}')
    else:
        # if loading from backup
        # bake_index = np.load("bake_index.bac.npy")
        # if 'bake_index' in h5file:
        #     del h5file['bake_index']
        # h5file.create_dataset(
        #         'bake_index',
        #         data=bake_index,
        # )
        bake_index = h5file['bake_index'][:]

    # PV
    if not skip_pv:
        dataset.pv.to_pickle("pv.pkl")

    # HRV
    if not skip_hrv:
        times = set(bake_index['time'][bake_index['hrv_flags'].any(axis=1)])
        times = sorted(list(times))

        if 'hrv' in h5file:
            del h5file['hrv']
        ds = h5file.create_dataset(
                "hrv",
                shape=(1, len(times), 12, 592, 684),
                dtype=np.uint8,
                chunks=(1, 8, 12, 592, 684),
        )
        ds.attrs['times'] = np.array(times, dtype=np.uint32)

        logger.info('Writing hrv data to hdf5 file')
        for i, timeint in enumerate(tqdm(times)):
            time = datetime.fromtimestamp(timeint)
            d = dataset.hrv['data'].sel(time=slice(str(time), str(time + timedelta(minutes=55)))).to_numpy()
            # t, y, x, c -> (c), t, y, x
            ds[:, i] = (d.clip(0, 1) * 255).astype(np.uint8).transpose(3, 0, 1, 2)
        h5file.flush()

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
                chunks=(1, 8, 12, 293, 333),
        )
        ds.attrs['times'] = np.array(times, dtype=np.uint32)

        logger.info('Writing nonhrv data to hdf5 file')
        for i, timeint in enumerate(tqdm(times)):
            time = datetime.fromtimestamp(timeint)
            d = dataset.nonhrv['data'].sel(time=slice(str(time), str(time + timedelta(minutes=55)))).to_numpy()
            # t, y, x, c -> c, t, y, x
            ds[:, i] = (d.clip(0, 1) * 255).astype(np.uint8).transpose(3, 0, 1, 2)
        h5file.flush()

    # WEATHER
    if not skip_weather:
        times = set()
        for timeint, _, _, _, weather_flags, _, _ in bake_index:
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

        logger.info('Writing weather data to hdf5 file')
        for i, timeint in enumerate(tqdm(times)):
            time = datetime.fromtimestamp(timeint)
            weather_data = dataset.weather.sel(time=time)
            weather_data = xr.concat([
                    (weather_data[k.name.lower()] - WEATHER_RANGES[k][0]) / (WEATHER_RANGES[k][1] - WEATHER_RANGES[k][0])
                    for k in WEATHER
                ], dim="channel").values
            # c, y, x -> c, y, x
            ds[:, i] = (weather_data.clip(0, 1) * 255).astype(np.uint8)
        h5file.flush()

    # AEROSOLS
    if not skip_aerosols:
        times = set()
        for timeint, _, _, _, _, aerosols_flags, _ in bake_index:
            if not aerosols_flags.any():
                continue
            for j in range(-1, 5):
                times.add(timeint + j * 3600)
        times = sorted(list(times))

        if 'aerosols' in h5file:
            del h5file['aerosols']
        ds = h5file.create_dataset(
                "aerosols",
                shape=(13, len(times), 8, 250, 240),
                dtype=np.uint8,
                chunks=(1, 8, 8, 250, 240),
        )
        ds.attrs['times'] = np.array(times, dtype=np.uint32)

        logger.info('Writing aerosols data to hdf5 file')
        for i, timeint in enumerate(tqdm(times)):
            time = datetime.fromtimestamp(timeint)
            aerosols_data = dataset.aerosols.sel(time=time)
            aerosols_data = xr.concat([
                    (aerosols_data[k.name.lower()] - AEROSOLS_RANGES[k][0]) / (AEROSOLS_RANGES[k][1] - AEROSOLS_RANGES[k][0])
                    for k in AEROSOLS
                ], dim="channel").values
            # c, h, y, x -> c, h, y, x
            ds[:, i] = (aerosols_data.clip(0, 1) * 255).astype(np.uint8)
        h5file.flush()

    h5file.close()


if __name__ == "__main__":
    main(
        overwrite       = False,
        skip_bake_index = False,
        skip_pv         = False,
        skip_hrv        = False,
        skip_nonhrv     = False,
        skip_weather    = False,
        skip_aerosols   = False,
    )
