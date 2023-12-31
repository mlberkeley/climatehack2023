import sys

sys.path.append("./")

from submission.config import config
from datetime import datetime
from data.data import ChallengeDataset
from data.random_data import ClimatehackDatasetConfig, ClimatehackDataset
from torch.utils.data import DataLoader
from pathlib import Path


N_ITER = 10


def benchmark(dataloader: iter):
    start = datetime.now()
    for i, (time, site, pv_features, hrv_features, pv_targets, _) in enumerate(dataloader):
        if i == N_ITER:
            break
        pass
    end = datetime.now()
    return (end - start) / N_ITER


def main():
    start = datetime.now()
    iter_dataset = ChallengeDataset("nonhrv", 2020)
    init_time = datetime.now() - start
    dataloader = DataLoader(iter_dataset, batch_size=config.train.batch_size, pin_memory=True)
    time = benchmark(dataloader)
    print(f'Iterative dataset: {init_time} + {time}/iter')

    start = datetime.now()
    ds_conf = ClimatehackDatasetConfig(
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        root_dir=Path("/data/climatehack/official_dataset"),
        features=None,
    )
    climatehack_dataset = ClimatehackDataset(ds_conf)
    init_time = datetime.now() - start
    dataloader = DataLoader(climatehack_dataset, batch_size=config.train.batch_size, pin_memory=True)
    time = benchmark(dataloader)
    print(f'Climatehack Dataset (workers=0): {init_time} + {time}/iter')

    dataloader = DataLoader(climatehack_dataset, batch_size=config.train.batch_size, pin_memory=True, num_workers=2)
    time = benchmark(dataloader)
    print(f'Climatehack Dataset (workers=2): {init_time} + {time}/iter')

    dataloader = DataLoader(climatehack_dataset, batch_size=config.train.batch_size, pin_memory=True, num_workers=4)
    time = benchmark(dataloader)
    print(f'Climatehack Dataset (workers=4): {init_time} + {time}/iter')

    dataloader = DataLoader(climatehack_dataset, batch_size=config.train.batch_size, pin_memory=True, num_workers=8)
    time = benchmark(dataloader)
    print(f'Climatehack Dataset (workers=8): {init_time} + {time}/iter')

    dataloader = DataLoader(climatehack_dataset, batch_size=config.train.batch_size, pin_memory=True, num_workers=16)
    time = benchmark(dataloader)
    print(f'Climatehack Dataset (workers=16): {init_time} + {time}/iter')


if __name__ == "__main__":
    main()
