from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pickle
import h5py
import submission.util
from submission.models.keys import KeyEnum, META, COMPUTED, HRV, NONHRV, WEATHER, AEROSOLS
from loguru import logger
from easydict import EasyDict
import torchvision.transforms as transforms
from tqdm import tqdm


# Think about whether all channels need to be flipped together or not
TRAIN_TRANSFORM = transforms.Compose([
    # transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=True),
    # transforms.RandomHorizontalFlip(p=0.5),
    # # transforms.RandomVerticalFlip(p=0.5),
    # transforms.RandomApply([
    #     transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    # ], p=0.25),
])


def get_dataloaders(
    config: EasyDict,
    features: set[KeyEnum],
    load_train: bool = True,
    load_eval: bool = True,
):
    assert load_train or load_eval, "At least one of load_train or load_eval must be True"
    ret = []
    if load_train:
        start_time = datetime.now()
        train_dataset = ClimatehackDataset(
            start_date=config.data.train_start_date,
            end_date=config.data.train_end_date,
            root_dir=config.data.root,
            features=features,
            subset_size=config.data.train_subset_size,
            transform=TRAIN_TRANSFORM,
        )
        logger.info(f"Loaded train dataset with {len(train_dataset):,} samples in {datetime.now() - start_time}")
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            pin_memory=True,
            num_workers=config.data.num_workers,
            shuffle=True,
        )
        ret.append(train_loader)
    if load_eval:
        start_time = datetime.now()
        eval_dataset = ClimatehackDataset(
            start_date=config.data.eval_start_date,
            end_date=config.data.eval_end_date,
            root_dir=config.data.root,
            features=features,
            subset_size=config.data.eval_subset_size,
        )
        logger.info(f"Loaded eval dataset with {len(eval_dataset):,} samples in {datetime.now() - start_time}")
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=config.eval.batch_size,
            pin_memory=True,
            num_workers=config.data.num_workers,
            shuffle=False,
        )
        ret.append(eval_loader)
    if len(ret) == 1:
        return ret[0]
    return ret


class ClimatehackDataset(Dataset):

    def __init__(self,
        start_date: datetime,
        end_date: datetime,
        root_dir: Path,
        features: set[KeyEnum],
        subset_size: int = 0,
        transform=None,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.root_dir = root_dir
        logger.debug(f"Loading dataset with features: {features}")
        self.meta_features = {k for k in features if META.has(k)}
        self.computed_features = {k for k in features if COMPUTED.has(k)}
        self.hrv_features = {k for k in features if HRV.has(k)}
        self.nonhrv_features = {k for k in features if NONHRV.has(k)}
        self.weather_features = {k for k in features if WEATHER.has(k)}
        self.aerosols_features = {k for k in features if AEROSOLS.has(k)}
        self.require_future_nonhrv = COMPUTED.FUTURE_NONHRV in features

        self.transform = transform if transform is not None else nn.Identity()

        self.meta = pd.read_csv("/data/climatehack/official_dataset/pv/meta.csv")

        datafile = h5py.File(f'{root_dir}/baked_data_v2.h5', 'r')

        # bake index
        start_time = datetime.now()
        self.bake_index = np.empty_like(datafile['bake_index'])
        datafile['bake_index'].read_direct(self.bake_index)
        self._filter_bake_index()
        if (subset_size > 0 and len(self.bake_index) > subset_size):
            rng = np.random.default_rng(21)
            rng.shuffle(self.bake_index)
            self.bake_index = self.bake_index[:subset_size]
        logger.debug(f"Loaded bake index in {datetime.now() - start_time}")

        # pv
        start_time = datetime.now()
        self.pv = pd.read_pickle(f"{root_dir}/pv.pkl")
        logger.debug(f"Loaded pv in {datetime.now() - start_time}")

        # hrv
        start_time = datetime.now()
        hrv_src = datafile['hrv']
        self.hrv, self.hrv_time_map = self._load_data(hrv_src, [ch.value for ch in self.hrv_features])
        logger.debug(f"Loaded hrv in {datetime.now() - start_time}")

        # nonhrv
        start_time = datetime.now()
        nonhrv_src = datafile['nonhrv']
        self.nonhrv, self.nonhrv_time_map = self._load_data(nonhrv_src, [ch.value for ch in self.nonhrv_features])
        logger.debug(f"Loaded nonhrv in {datetime.now() - start_time}")

        # weather
        start_time = datetime.now()
        weather_src = datafile['weather']
        self.weather, self.weather_time_map = self._load_data(weather_src, [ch.value for ch in self.weather_features])
        logger.debug(f"Loaded weather in {datetime.now() - start_time}")

        # aerosols
        start_time = datetime.now()
        aerosols_src = datafile['aerosols']
        self.aerosols, self.aerosols_time_map = self._load_data(aerosols_src, [ch.value for ch in self.aerosols_features])
        logger.debug(f"Loaded aerosols in {datetime.now() - start_time}")

        # TODO move this to data.h5
        with open(f'{root_dir}/indices.json') as f:
            self.site_locations = {
                data_source: {
                    int(site): (int(location[0]), int(location[1]))
                    for site, location in locations.items()
                } for data_source, locations in json.load(f).items()
            }

        datafile.close()

    def _load_data(self, src: h5py.Dataset, channels: list[int], start_time: datetime = None, end_time: datetime = None):
        if start_time is None:
            start_time = self.start_date
        if end_time is None:
            end_time = self.end_date
        times = src.attrs['times']
        start_i = np.argmax(times >= start_time.timestamp())
        end_i = times.shape[0] - np.argmax(times[::-1] < end_time.timestamp())

        output = np.empty(
                (len(channels), end_i - start_i, *src.shape[2:]),
                dtype=np.uint8
        )
        # this does: out = src[selected_channels, selected_times] but faster
        # (unless selected channels are consequtive then it's not faster, but let's ignore that tiny detail)
        for i, ch in enumerate(channels):
            src.read_direct(output[i], np.s_[ch, start_i:end_i])

        output.setflags(write=False)
        time_map = {t: i for i, t in enumerate(times[start_i:end_i])}

        return output, time_map

    def _filter_bake_index(self):
        # TODO  only need to check required features
        self.bake_index = self.bake_index[
            (self.bake_index['time'] >= self.start_date.timestamp()) &
            (self.bake_index['time'] < self.end_date.timestamp()) &
            self.bake_index['hrv_flags'].all(axis=1) &
            self.bake_index['nonhrv_flags'].all(axis=1) &
            self.bake_index['weather_flags'].all(axis=1) &
            self.bake_index['aerosols_flags'].all(axis=1)
        ]
        # TODO  also check for future hrv
        if self.require_future_nonhrv:
            logger.info("Filtering bake index for future nonhrv.")
            keep_map = np.zeros(len(self.bake_index), dtype=bool)
            # basically need it so that for every site, there's nonhrv for hour and next 4 hrs
            # for entry in bake_index:
            #     check if bakeindex[site, time + 1:time + 5] all exist in bakeindex
            #     if so, keep_map[entry] = True
            # self.bake_index = self.bake_index[keep_map]
            bake_index_entries = {
                (ts, s) for ts, s, _, _, _, _, _ in self.bake_index
            }
            def _check_future_nonhrv(timestamp, site):
                for i in range(1, 5):
                    if (timestamp + i * 3600, site) not in bake_index_entries:
                        return False
                return True

            for i, (timestamp, site, _, _, _, _, _) in enumerate(tqdm(self.bake_index)):
                keep_map[i] = _check_future_nonhrv(timestamp, site)
            self.bake_index = self.bake_index[keep_map]

    def __len__(self):
        return len(self.bake_index)

    def __getitem__(self, idx):
        timestamp, site, hrv_flags, nonhrv_flags, weather_flags, aerosols_flags, solar_cache = self.bake_index[idx]
        time = datetime.fromtimestamp(timestamp)

        # pv
        first_hour = slice(
            str(time),
            str(time + timedelta(minutes=55))
        )
        next_four = slice(
            str(time + timedelta(hours=1)),
            str(time + timedelta(hours=4, minutes=55)),
        )
        pv_features = self.pv.xs(first_hour).xs(site).to_numpy().squeeze(-1)
        pv_targets = self.pv.xs(next_four).xs(site).to_numpy().squeeze(-1)

        out = {}

        # hrv
        x, y = self.site_locations['hrv'][site]
        hrv_ind = self.hrv_time_map[timestamp]
        hrv_out_raw = self.hrv[
                :,
                hrv_ind,
                :,
                y - 64:y + 64,
                x - 64:x + 64,
        ]
        hrv_out_raw = hrv_out_raw.astype(np.float32) / 255
        for i, key in enumerate(self.hrv_features):
            out[key] = self.transform(torch.from_numpy(hrv_out_raw[i]))

        # nonhrv
        x, y = self.site_locations['nonhrv'][site]
        if not self.require_future_nonhrv:
            nonhrv_ind = self.nonhrv_time_map[timestamp]
            nonhrv_out_raw = self.nonhrv[
                    :,
                    nonhrv_ind,
                    :,
                    y - 64:y + 64,
                    x - 64:x + 64,
            ]
        else:
            nonhrv_inds = [self.nonhrv_time_map[timestamp + i * 3600] for i in range(5)]
            nonhrv_out_raw = self.nonhrv[
                    :,
                    nonhrv_inds,
                    :,
                    y - 64:y + 64,
                    x - 64:x + 64,
            ]
            nonhrv_out_raw = nonhrv_out_raw.reshape(-1, 60, 128, 128)

        # nonhrv.shape = (channels, num_hours, hour, y, x) = (11, *, 12, 293, 333)
        # nonhrv_out_raw.shape = (len(nonhrv_keys), 12, 128, 128)
        nonhrv_out_raw = nonhrv_out_raw.astype(np.float32) / 255
        for i, key in enumerate(self.nonhrv_features):
            out[key] = self.transform(torch.from_numpy(nonhrv_out_raw[i]))

        # weather
        x, y = self.site_locations['weather'][site]
        weather_ind = self.weather_time_map[timestamp]
        weather_out_raw = self.weather[
                :,
                weather_ind - 1:weather_ind + 5,
                y - 64:y + 64,
                x - 64:x + 64,
        ]
        # weather.shape = (channels, num_hours, y, x) = (38, *, 305, 289)
        # weather_out_raw.shape = (len(weather_keys), 6, 128, 128)
        weather_out_raw = weather_out_raw.astype(np.float32) / 255
        for i, key in enumerate(self.weather_features):
            out[key] = self.transform(torch.from_numpy(weather_out_raw[i]))

        # aerosols
        x, y = self.site_locations['aerosols'][site]
        aerosols_ind = self.aerosols_time_map[timestamp]
        aerosols_out_raw = self.aerosols[
                :,
                aerosols_ind,
                :,
                y - 64:y + 64,
                x - 64:x + 64,
        ]
        aerosols_out_raw = aerosols_out_raw.astype(np.float32) / 255
        for i, key in enumerate(self.aerosols_features):
            out[key] = self.transform(torch.from_numpy(aerosols_out_raw[i]))

        # meta
        df = self.meta
        ss_id, lati, longi, _, orientation, tilt, kwp, _ = df.iloc[np.searchsorted(df['ss_id'].values, site)]
        out[META.TIME] = timestamp
        out[META.LATITUDE] = lati
        out[META.LONGITUDE] = longi
        out[META.ORIENTATION] = orientation
        out[META.TILT] = tilt
        out[META.KWP] = kwp
        out[COMPUTED.SOLAR_ANGLES] = solar_cache

        return pv_features, out, pv_targets
