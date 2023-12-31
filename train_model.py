from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
import wandb
import argparse

from data.data import ChallengeDataset
from data.random_data import ClimatehackDataset
from submission.resnet import ResNet18 as Model
from submission.config import config
from util import util
from eval import eval
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--run_name", type=str, default=None)
parser.add_argument("-m", "--run_notes", type=str, default=None)

args = parser.parse_args()

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_save_name = config.train.model_save_name

if device == "cpu":
    print("====================")
    for i in range(10):
        print("YOU ARE IN CPU MODE")


# summary(Model(), input_size=[(1, 12), (1, 12, 128, 128), (1, 6 * len(config.train.weather_keys), 128, 128)], device=device)
summary(Model(), input_size=[(1, 12), (1, 12, 128, 128)], device=device)

validation_loss, min_val_loss = 0, .15

# Actually do the training wow
model = Model().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.train.lr)

# dataset = ChallengeDataset(data, year)
# dataloader = DataLoader(dataset, batch_size=config.train.batch_size, pin_memory=True)
start = datetime.now()
dataset = ClimatehackDataset(
        start_date=config.data.train_start_date,
        end_date=config.data.train_end_date,
        root_dir=Path("/data/climatehack/"),
        features=None,
        subset_size=config.data.train_subset_size,
)
print(f'Train dataset length: {len(dataset):,}, loaded in {datetime.now() - start}')
dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        pin_memory=True,
        num_workers=config.data.num_workers,
        shuffle=True
)

# eval_dataset = ChallengeDataset(data, 2021, eval=True, eval_year=2021, eval_day=15, eval_hours=96)
# eval_loader = DataLoader(eval_dataset, batch_size=config.train.batch_size, pin_memory=True)
start = datetime.now()
eval_dataset = ClimatehackDataset(
        start_date=config.data.eval_start_date,
        end_date=config.data.eval_end_date,
        root_dir=Path("/data/climatehack/"),
        features=None,
        subset_size=config.data.eval_subset_size,
)
print(f'Eval dataset length: {len(eval_dataset):,}, loaded in {datetime.now() - start}')
eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.eval.batch_size,
        pin_memory=True,
        num_workers=config.data.num_workers,
        shuffle=False
)

wandb.init(
    entity="mlatberkeley",
    project="climatehack23",
    config=dict(config),
    mode="online",
    name=args.run_name,
    notes=args.run_notes
)

for epoch in range(config.train.num_epochs):
    print(f"[{datetime.now()}]: Epoch {epoch + 1}")
    model.train()

    running_loss = 0.0
    count = 0
    last_time = datetime.now()
    for i, (time, site, pv_features, pv_targets, nonhrv_features, nwp_features) in enumerate(dataloader):
        optimizer.zero_grad()
        pv_features, nonhrv_features, nwp_features, pv_targets = pv_features.to(device, dtype=torch.float), nonhrv_features.to(device, dtype=torch.float), nwp_features.to(device, dtype=torch.float), pv_targets.to(device, dtype=torch.float)

        predictions = model(
            pv_features,
            nonhrv_features,
            # nwp_features,
        )

        loss = criterion(predictions, pv_targets.to(device, dtype=torch.float))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad_norm)
        optimizer.step()

        size = int(pv_targets.size(0))
        running_loss += float(loss) * size
        count += size

        if i % 10 == 6:
            print(f"Epoch {epoch + 1}, {i + 1}: loss: {running_loss / count}, time: {time[0]}")
            os.makedirs("submission", exist_ok=True)
            torch.save(model.state_dict(), f"submission/{file_save_name}")

            sample_pv, sample_vis = util.visualize_example(
                pv_features[0], pv_targets[0], predictions[0], nonhrv_features[0]
            )

            if i % 80 == 6:
                st = datetime.now()
                print(f"validating: start {datetime.now()}")
                validation_loss = eval(eval_dataloader, model)
                print(f"loss: {validation_loss}, validation time {datetime.now() - st}")
                if validation_loss < min_val_loss:
                    torch.save(model.state_dict(), f"submission/best_{file_save_name}")
                    min_val_loss = validation_loss

            wandb.log({
                "train_loss": running_loss / (count + 1e-10),
                "validation_loss": validation_loss,
                "sample_pv": sample_pv,
                "sample_vis": sample_vis,
            })
        # if i % 25 == 0:
        #     print(f'iter time: {(datetime.now() - last_time) / 25}')
        #     last_time = datetime.now()


    print(f"Epoch {epoch + 1}: {running_loss / (count + 1e-10)}")

# Save your model
os.makedirs("submission", exist_ok=True)
torch.save(model.state_dict(), f"submission/{file_save_name}")

wandb.finish()
