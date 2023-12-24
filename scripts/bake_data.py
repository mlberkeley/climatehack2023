import sys

sys.path.append("./")

from datetime import datetime
import numpy as np

from data.data import ChallengeDataset


def main():
    start = datetime.now()

    dataset = ChallengeDataset("nonhrv", 2021)

    baked_goods = []
    for i, (date, site, _, _, _) in enumerate(iter(dataset)):
        if i % 1000 == 0:
            print(i)
        baked_goods.append((date, site))

    baked_goods = np.array(baked_goods, dtype=np.uint32)
    np.save("baked_nonhrv_2021.npy", baked_goods)

    end = datetime.now()
    print(f'Time elapsed: {end - start}')


if __name__ == "__main__":
    main()
