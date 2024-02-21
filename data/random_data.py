from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pickle
import h5py
from submission.util import util
import submission.keys as keys
from loguru import logger
from easydict import EasyDict
import torchvision.transforms as transforms
from tqdm import tqdm


# Think about whether all channels need to be flipped together or not
TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    # transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=True),
    # transforms.RandomHorizontalFlip(p=0.5),
    # # transforms.RandomVerticalFlip(p=0.5),
    # transforms.RandomApply([
    #     transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    # ], p=0.25),
])


def get_dataloaders(
    config: EasyDict,
    meta_features: set[keys.META],
    nonhrv_features: set[keys.NONHRV],
    weather_features: set[keys.WEATHER],
    future_features: set[keys.FUTURE],
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
            meta_features=meta_features,
            nonhrv_features=nonhrv_features,
            weather_features=weather_features,
            future_features=future_features,
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
            meta_features=meta_features,
            nonhrv_features=nonhrv_features,
            weather_features=weather_features,
            future_features=future_features,
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
        meta_features: set[keys.META],
        nonhrv_features: set[keys.NONHRV],
        weather_features: set[keys.WEATHER],
        future_features: set[keys.FUTURE],
        subset_size: int = 0,
        transform=None,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.root_dir = root_dir
        self.meta_features = meta_features
        self.nonhrv_features = nonhrv_features
        self.weather_features = weather_features
        self.require_future_nonhrv = keys.FUTURE.NONHRV in future_features if future_features else False
        self.transform = transform if transform is not None else transforms.ToTensor()

        self.meta = pd.read_csv("/data/climatehack/official_dataset/pv/meta.csv")

        datafile = h5py.File(f'{root_dir}/baked_data.h5', 'r')

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

        # nonhrv
        start_time = datetime.now()
        nonhrv_src = datafile['nonhrv']
        # output dim (len(nonhrv_features), end_i - start_i, *src.shape[2:])
        self.nonhrv, self.nonhrv_time_map = self._load_data(
                nonhrv_src,
                [ch.value for ch in self.nonhrv_features],
                start_date,
                end_date,
        )
        logger.debug(f"Loaded nonhrv in {datetime.now() - start_time}")

        # weather
        start_time = datetime.now()
        weather_src = datafile['weather']
        self.weather, self.weather_time_map = self._load_data(
                weather_src,
                [ch.value for ch in self.weather_features],
                start_date,
                end_date,
        )
        logger.debug(f"Loaded weather in {datetime.now() - start_time}")

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
        # (unless selected channels are consequtive then it's not faster, but let's ignore that tiny detail)
        for i, ch in enumerate(channels):
            src.read_direct(output[i], np.s_[ch, start_i:end_i])

        time_map = {t: i for i, t in enumerate(times[start_i:end_i])}
        return output, time_map

    def _filter_bake_index(self):
        # TODO  only need to check required features
        self.bake_index = self.bake_index[
            (self.bake_index['time'] >= self.start_date.timestamp()) &
            (self.bake_index['time'] < self.end_date.timestamp()) &
            self.bake_index['nonhrv_flags'].all(axis=1) &
            self.bake_index['weather_flags'].all(axis=1)
        ]
        # TODO  use self.require_future_nonhrv
        if self.require_future_nonhrv:
            keep_map = np.zeros(len(self.bake_index), dtype=bool)
            # basically need it so that for every site, there's nonhrv for hour and next 4 hrs
            # for entry in bake_index:
            #     check if bakeindex[site, time + 1:time + 5] all exist in bakeindex
            #     if so, keep_map[entry] = True
            # self.bake_index = self.bake_index[keep_map]
            bake_index_entries = {
                (ts, s) for ts, s, _, _ in self.bake_index
            }
            def _check_future_nonhrv(timestamp, site):
                for i in range(1, 5):
                    if (timestamp + i * 3600, site) not in bake_index_entries:
                        return False
                return True

            for i, (timestamp, site, _, _) in enumerate(tqdm(self.bake_index)):
                keep_map[i] = _check_future_nonhrv(timestamp, site)
            self.bake_index = self.bake_index[keep_map]

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
        next_four = slice(
            str(time + timedelta(hours=1)),
            str(time + timedelta(hours=4, minutes=55)),
        )
        pv_features = self.pv.xs(first_hour).xs(site).to_numpy().squeeze(-1)
        pv_targets = self.pv.xs(next_four).xs(site).to_numpy().squeeze(-1)

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
        # toTensor expects (C, H, W) but we have (H, W, C)
        nonhrv_out_raw = np.swapaxes(nonhrv_out_raw, 1, -1)
        nonhrv_out = {
            key: self.transform(nonhrv_out_raw[i])
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
        weather_out_raw = weather_out_raw.astype(np.float32) / 255
        # toTensor expects (C, H, W) but we have (H, W, C)
        weather_out_raw = np.swapaxes(weather_out_raw, 1, -1)
        weather_out = {
            key: self.transform(weather_out_raw[i])
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

        return pv_features, meta_out, nonhrv_out, weather_out, pv_targets
