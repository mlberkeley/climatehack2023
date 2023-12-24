import sys

sys.path.append("./")

from submission.config import config
import datetime
from data.data import ChallengeDataset
from torch.utils.data import DataLoader


N_ITER = 10


def benchmark(dataloader: iter):
    start = datetime.datetime.now()
    for i, (time, site, pv_features, hrv_features, pv_targets) in enumerate(dataloader):
        if i == N_ITER:
            break
        pass
    end = datetime.datetime.now()
    return end - start


def main():
    iter_dataset = ChallengeDataset("nonhrv", 2020)
    dataloader = DataLoader(iter_dataset, batch_size=config.train.batch_size, pin_memory=True)
    time = benchmark(dataloader)
    print(f'ChallengeDataset: {time}')


if __name__ == "__main__":
    main()

