import os
import sys
from pathlib import Path

from loguru import logger
from datetime import datetime
import numpy as np

import wandb
import argparse
from config import get_config

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
from ema_pytorch import EMA

from data.random_data import get_dataloaders
import submission.util as util

from submission.models.keys import META, COMPUTED, HRV, NONHRV, WEATHER, AEROSOLS
from submission.models import build_model


logger.info("imported modules")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument("--nowandb", action='store_true')
parser.add_argument("-n", "--run_name", type=str, required=True)
parser.add_argument("-t", "--run_type", type=str, choices=["train", "eval"], default="train")
parser.add_argument("-m", "--run_notes", type=str, default=None)
parser.add_argument("-g", "--run_group", type=str, default=None)
parser.add_argument("-c", "--config", type=str, required=True, help='filepath for config to use')
parser.add_argument('--opts', nargs='*', default=[], help='arguments to override config as key=value')

args = parser.parse_args()
config = get_config(args.config, args.opts)


def eval(dataloader, model, criterion=nn.L1Loss(), preds_save_path=None, ground_truth_path=None):
    model.eval()

    tot_loss_1h, tot_loss_4h, count = 0, 0, 0

    gt = np.zeros((len(dataloader.dataset), 48))
    preds = np.zeros((len(dataloader.dataset), 48))

    logger.info("started eval")

    with torch.no_grad():
        for i, (pv_features, features, pv_targets) in enumerate(dataloader):
            features = util.dict_to_device(features)
            pv_features = pv_features.to(device, dtype=torch.float)
            pv_targets = pv_targets.to(device, dtype=torch.float)

            predictions = model(pv_features, features)

            gt[i * dataloader.batch_size: (i + 1) * dataloader.batch_size] = pv_targets.cpu().numpy()
            preds[i * dataloader.batch_size: (i + 1) * dataloader.batch_size] = predictions.cpu().numpy()

            loss_1h = criterion(predictions[:, :12], pv_targets[:, :12])
            loss_4h = criterion(predictions, pv_targets)

            size = int(pv_targets.size(0))
            tot_loss_1h += float(loss_1h) * size
            tot_loss_4h += float(loss_4h) * size
            count += size

    logger.info("finished eval")

    model.train()

    val_loss_1h = tot_loss_1h / count
    val_loss_4h = tot_loss_4h / count

    if preds_save_path is not None:
        np.save(preds_save_path, preds)

    if ground_truth_path is not None:
        np.save(ground_truth_path, gt)

    return val_loss_1h, val_loss_4h


def train():
    os.makedirs(f'ckpts/{args.run_name}/', exist_ok=True)
    save_path = f'ckpts/{args.run_name}/model.pt'

    util.save_config_to_json(config, f'ckpts/{args.run_name}/config.json')

    if Path(save_path).exists() or Path(save_path + '.best').exists():
        logger.error(f"Model save path {save_path} already exists, exiting. ")
        logger.error("Rename the run or delete the existing checkpoint to continue.")
        exit()

    if device == "cpu":
        logger.critical('CPU mode is not supported (well it is but it is very slow)')
        exit()

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        logger.warning('CUDA_VISIBLE_DEVICES not set, ensure you are using a free GPU')


    model = build_model(config).to(device)

    train_dataloader, eval_dataloader = get_dataloaders(
            config=config,
            features=model.required_features
    )

    features = model.required_features
    meta_features = {k for k in features if META.has(k)}
    computed_features = {k for k in features if COMPUTED.has(k)}
    hrv_features = {k for k in features if HRV.has(k)}
    nonhrv_features = {k for k in features if NONHRV.has(k)}
    weather_features = {k for k in features if WEATHER.has(k)}
    aerosols_features = {k for k in features if AEROSOLS.has(k)}
    require_future_nonhrv = COMPUTED.FUTURE_NONHRV in features
    summary(model, input_data=(
        torch.zeros((1, 12)),
        {k: torch.zeros((1, )) for k in meta_features} |
        {COMPUTED.SOLAR_ANGLES: torch.zeros((1, 6, 2))} |
        {k: torch.zeros((1, 12, 128, 128)) for k in hrv_features} |
        {k: torch.zeros((1, 60 if require_future_nonhrv  else 12, 128, 128)) for k in nonhrv_features} |
        {k: torch.zeros((1, 6, 128, 128)) for k in weather_features} |
        {k: torch.zeros((1, 6, 128, 128)) for k in aerosols_features},
    ), device=device)


    wandb.init(
        entity="mlatberkeley",
        project="climatehack23",
        config=dict(config),
        mode="offline" if args.nowandb else "online",
        name=args.run_name,
        notes=args.run_notes,
        group=args.run_group,
    )

    ema = EMA(
        model,
        beta = 0.999,               # exponential moving average factor
        update_after_step = 100,    # only after this number of .update() calls will it start updating
        update_every = 10,           # how often to actually update, to save on compute (updates every 10th .update() call)
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.train.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    torch.autograd.set_detect_anomaly(True)

    val_loss_1h, val_loss_4h = eval(eval_dataloader, model)
    ema_loss_1h, ema_loss_4h = val_loss_1h, val_loss_4h
    min_val_loss = val_loss_4h
    min_ema_loss = ema_loss_4h


    logger.info("started training")
    for epoch in range(config.train.num_epochs):
        logger.info(f"[{datetime.now()}]: Epoch {epoch + 1}")
        model.train()

        running_losses = {
            'loss': 0,
            'l1_1h': 0,
            'l1_4h': 0,
        }

        count = 0
        last_time = datetime.now()

        for i, (pv_features, features, pv_targets) in enumerate(train_dataloader):
            optimizer.zero_grad()

            features = util.dict_to_device(features)
            pv_features = pv_features.to(device, dtype=torch.float)
            pv_targets = pv_targets.to(device, dtype=torch.float)

            predictions = model(pv_features, features)

            loss = criterion(predictions, pv_targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad_norm)
            optimizer.step()
            ema.update()

            size = int(pv_targets.size(0))
            running_losses['loss'] += float(loss) * size
            running_losses['l1_1h'] += float(F.l1_loss(predictions[:, :12], pv_targets[:, :12])) * size
            running_losses['l1_4h'] += float(F.l1_loss(predictions, pv_targets)) * size
            count += size

            if (i + 1) % config.train.eval_every == 0:
                st = datetime.now()
                logger.info(f"validating...")
                val_loss_1h, val_loss_4h = eval(eval_dataloader, model)
                ema_loss_1h, ema_loss_4h = eval(eval_dataloader, ema)
                logger.info(f"val_l1 - 1h: {val_loss_1h:.5f}, 4h: {val_loss_4h:.5f}, ema_l1 - 1h: {ema_loss_1h:.5f}, 4h: {ema_loss_4h:.5f}")

                torch.save(model.state_dict(), save_path)
                if val_loss_4h < min_val_loss:
                    torch.save(model.state_dict(), save_path + '.best')
                    min_val_loss = val_loss_4h
                if ema_loss_4h < min_ema_loss:
                    torch.save(ema.ema_model.state_dict(), save_path + '.best_ema')
                    min_ema_loss = ema_loss_4h

            if (i + 1) % config.train.wandb_log_every == 0:
                #sample_pv, sample_vis = util.visualize_example(
                    #pv_features[0], pv_targets[0], predictions[0], nonhrv_features[0]
                #)
                wandb.log({
                    'train_loss': running_losses['loss'] / count,
                    'train_l1_1h': running_losses['l1_1h'] / count,
                    'train_l1_4h': running_losses['l1_4h'] / count,
                    'val_loss_1h': val_loss_1h,
                    'val_loss_4h': val_loss_4h,
                    'ema_val_loss_1h': ema_loss_1h,
                    'ema_val_loss_4h': ema_loss_4h,
                    # "lr": scheduler.get_last_lr()[0],
                    #"sample_pv": sample_pv,
                    #"sample_vis": sample_vis,
                })

            if (i + 1) % config.train.log_every == 0:
                logger.info(
                        f"Epoch {epoch + 1}, {i + 1}: "
                        f"loss: {running_losses['loss'] / count:.5f}, "
                        f"l1_1h: {running_losses['l1_1h'] / count:.5f}, "
                        f"l1_4h: {running_losses['l1_4h'] / count:.5f}"
                )
                running_losses = dict.fromkeys(running_losses, 0)
                count = 0

        scheduler.step()

        logger.info(f"Epoch {epoch + 1}: {running_losses['loss'] / count}")
        logger.info(f"LR: {scheduler.get_last_lr()} -> {scheduler.get_lr()}")

    wandb.finish()


if __name__ == "__main__":
    logger.info(f"started with {args.run_type}")

    if args.run_type == "train":
        train()
    else:
        # INFO: eval
        model = build_model(config).to(device)
        model.load_state_dict(torch.load(f'ckpts/{args.run_name}/model.pt'))
        model.eval()
        dataloader = get_dataloaders(
            config=config,
            meta_features=model.REQUIRED_META,
            hrv_features=model.REQUIRED_HRV,
            nonhrv_features=model.REQUIRED_NONHRV,
            weather_features=model.REQUIRED_WEATHER,
            aerosols_features=model.REQUIRED_AEROSOLS,
            future_features=None,
            load_train=False,
        )
        val_loss_1h, val_loss_4h = eval(
                dataloader, model,
                # preds_save_path=args.output, ground_truth_path=args.ground_truth
        )
        logger.info(f"val_l1 - 1h: {val_loss_1h:.5f}, 4h: {val_loss_4h:.5f}")
