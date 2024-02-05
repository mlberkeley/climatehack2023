import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import argparse

from torch.utils.data import DataLoader
from torchinfo import summary
from datetime import datetime

from data.random_data import get_dataloaders
import submission.keys as keys
from submission.resnet import ResNetPV as Model
from config import config
from util import util
from eval import eval
from pathlib import Path
from loguru import logger


# INFO: setup
parser = argparse.ArgumentParser()
parser.add_argument("--nowandb", action='store_true')
parser.add_argument("-n", "--run_name", type=str, default=None)
parser.add_argument("-m", "--run_notes", type=str, default=None)

args = parser.parse_args()

os.makedirs("submission", exist_ok=True)


torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == "cpu":
    logger.critical('CPU mode is not supported (well it is but it is very slow)')
    exit()
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    logger.warning('CUDA_VISIBLE_DEVICES not set, ensure you are using a free GPU')


model = Model().to(device)

summary(model, input_data=(
    torch.zeros((1, 12)),
    {k: torch.zeros((1, )) for k in model.REQUIRED_META},
    {k: torch.zeros((1, 12, 128, 128)) for k in model.REQUIRED_NONHRV},
    {k: torch.zeros((1, 6, 128, 128)) for k in model.REQUIRED_WEATHER},
), device=device)


train_dataloader, eval_dataloader = get_dataloaders(
        config=config,
        meta_features=model.REQUIRED_META,
        nonhrv_features=model.REQUIRED_NONHRV,
        weather_features=model.REQUIRED_WEATHER,
)

wandb.init(
    entity="mlatberkeley",
    project="climatehack23",
    config=dict(config),
    mode="offline" if args.nowandb else "online",
    name=args.run_name,
    notes=args.run_notes
)

# INFO: train
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.train.lr)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

val_loss_1h, val_loss_4h = eval(eval_dataloader, model)
min_val_loss = val_loss_4h

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
    for i, (pv_features, meta, nonhrv, weather, pv_targets) in enumerate(train_dataloader):
        optimizer.zero_grad()

        meta, nonhrv, weather = util.dict_to_device(meta), util.dict_to_device(nonhrv), util.dict_to_device(weather)
        pv_features = pv_features.to(device, dtype=torch.float)
        pv_targets = pv_targets.to(device, dtype=torch.float)

        predictions = model(pv_features, meta, nonhrv, weather)

        loss = criterion(predictions, pv_targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad_norm)
        optimizer.step()

        size = int(pv_targets.size(0))
        running_losses['loss'] += float(loss) * size
        running_losses['l1_1h'] += float(F.l1_loss(predictions[:, :12], pv_targets[:, :12])) * size
        running_losses['l1_4h'] += float(F.l1_loss(predictions, pv_targets)) * size
        count += size

        if (i + 1) % config.train.eval_every == 0:
            st = datetime.now()
            logger.info(f"validating...")
            val_loss_1h, val_loss_4h = eval(eval_dataloader, model)
            logger.info(f"val_l1 - 1h: {val_loss_1h:.5f}, 4h: {val_loss_4h:.5f}")

            torch.save(model.state_dict(), f"submission/{config.train.model_save_name}")
            if val_loss_4h < min_val_loss:
                torch.save(model.state_dict(), f"submission/best_{config.train.model_save_name}")
                min_val_loss = val_loss_4h

        if (i + 1) % config.train.log_every == 0:
            logger.info(
                    f"Epoch {epoch + 1}, {i + 1}: "
                    f"loss: {running_losses['loss'] / count:.5f}, "
                    f"l1_1h: {running_losses['l1_1h'] / count:.5f}, "
                    f"l1_4h: {running_losses['l1_4h'] / count:.5f}"
            )
            #sample_pv, sample_vis = util.visualize_example(
                #pv_features[0], pv_targets[0], predictions[0], nonhrv_features[0]
            #)
            wandb.log({
                "train_loss": running_losses['loss'] / count,
                "train_l1_1h": running_losses['l1_1h'] / count,
                "train_l1_4h": running_losses['l1_4h'] / count,
                "val_loss_1h": val_loss_1h,
                "val_loss_4h": val_loss_4h,
                #"sample_pv": sample_pv,
                #"sample_vis": sample_vis,
            })
            running_losses = dict.fromkeys(running_losses, 0)
            count = 0

    scheduler.step()

    logger.info(f"Epoch {epoch + 1}: {running_losses['loss'] / count}")
    logger.info(f"LR: {scheduler.get_last_lr()} -> {scheduler.get_lr()}")


wandb.finish()
