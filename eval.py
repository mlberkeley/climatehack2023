from config import get_config
from data.random_data import get_dataloaders
from submission.resnet import ResNetPV as Model
import submission.util

import argparse
from loguru import logger
import numpy as np
import torch
import torch.nn as nn


def eval(dataloader, model, criterion=nn.L1Loss(), preds_save_path=None, ground_truth_path=None):
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tot_loss_1h, tot_loss_4h, count = 0, 0, 0

    gt = np.zeros((len(dataloader.dataset), 48))
    preds = np.zeros((len(dataloader.dataset), 48))
    with torch.no_grad():
        for i, (pv_features, meta, nonhrv, weather, pv_targets) in enumerate(dataloader):
            meta, nonhrv, weather = util.dict_to_device(meta), util.dict_to_device(nonhrv), util.dict_to_device(weather)
            pv_features = pv_features.to(device, dtype=torch.float)
            pv_targets = pv_targets.to(device, dtype=torch.float)

            predictions = model(pv_features, meta, nonhrv, weather)

            gt[i * dataloader.batch_size: (i + 1) * dataloader.batch_size] = pv_targets.cpu().numpy()
            preds[i * dataloader.batch_size: (i + 1) * dataloader.batch_size] = predictions.cpu().numpy()

            loss_1h = criterion(predictions[:, :12], pv_targets[:, :12])
            loss_4h = criterion(predictions, pv_targets)

            size = int(pv_targets.size(0))
            tot_loss_1h += float(loss_1h) * size
            tot_loss_4h += float(loss_4h) * size
            count += size

    model.train()

    val_loss_1h = tot_loss_1h / count
    val_loss_4h = tot_loss_4h / count

    if preds_save_path is not None:
        np.save(preds_save_path, preds)
    if ground_truth_path is not None:
        np.save(ground_truth_path, gt)

    return val_loss_1h, val_loss_4h


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, required=True, help='name of the model')
    parser.add_argument('-o', '--output', type=str, default=None, help='path to save predictions')
    parser.add_argument('-gt', '--ground_truth', type=str, default=None, help='dir to save ground truth data')
    parser.add_argument("-c", "--config", type=str, required=True, help='config file to use')
    parser.add_argument('--opts', nargs='*', default=[], help='arguments to override config as key=value')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config(args.config, args.opts)

    model = Model(config.model.config).to(device)
    model.load_state_dict(torch.load(f'{args.name}'))
    model.eval()
    dataloader = get_dataloaders(
        config=config,
        meta_features=model.REQUIRED_META,
        nonhrv_features=model.REQUIRED_NONHRV,
        weather_features=model.REQUIRED_WEATHER,
        future_features=None,
        load_train=False,
    )
    val_loss_1h, val_loss_4h = eval(dataloader, model, preds_save_path=args.output, ground_truth_path=args.ground_truth)
    logger.info(f"val_l1 - 1h: {val_loss_1h:.5f}, 4h: {val_loss_4h:.5f}")
