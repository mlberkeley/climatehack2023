import os

import h5py
import numpy as np
import shutil
import sys

from submission.run import Evaluator

DATA_PATH = "/data/climatehack/evaluation/data.hdf5"


def main(model_folder):
    # Load the data (combined features & targets)
    try:
        data = h5py.File(DATA_PATH, "r")
    except FileNotFoundError:
        print(f"Unable to load features at `{DATA_PATH}`")
        return

    shutil.copyfile(f"{model_folder}/config.json", "submission/config.json")
    shutil.copyfile(f"{model_folder}/model.pt.best_ema", "submission/model.pt")
    # Switch into the submission directory
    cwd = os.getcwd()
    os.chdir("submission")
    

    # Make predictions on the data
    try:
        evaluator = Evaluator()

        predictions = []
        for batch in evaluator.predict(features=data):
            assert batch.shape[-1] == 48
            predictions.append(batch)
    finally:
        os.chdir(cwd)

    # Output the mean absolute error
    mae = np.mean(np.absolute(data["targets"] - np.concatenate(predictions)))
    print("MAE:", mae)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python doxa_local.py <model_folder>")
        exit(1)
    model_folder = sys.argv[1]
    main(model_folder)
