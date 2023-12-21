from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
import wandb

from data import ChallengeDataset
# from submission.model import Model
from submission.resnet import Model
from submission.config import config
from util import util
from eval import eval

wandb.init(
    entity="mlatberkeley",
    project="climatehack23",
    config=dict(config)
)

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == "cpu":
    print("====================")
    for i in range(10):
        print("YOU ARE IN CPU MODE")


summary(Model(), input_size=[(1, 12), (1, 6 * len(config.train.weather_keys), 128, 128)])
data = "nonhrv"
year = 2020
validation_loss = 0

# Actually do the training wow
model = Model().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.train.lr)

dataset = ChallengeDataset(data, year)
dataloader = DataLoader(dataset, batch_size=config.train.batch_size, pin_memory=True)

eval_dataset = ChallengeDataset(data, 2021, eval=True, eval_year=2021, eval_day=15, eval_hours=24)
eval_loader = DataLoader(eval_dataset, batch_size=config.train.batch_size, pin_memory=True)

for epoch in range(config.train.num_epochs):
    print(f"[{datetime.now()}]: Epoch {epoch + 1}")
    model.train()

    running_loss = 0.0
    count = 0
    for i, (time, site, pv_features, pv_targets, nwp_features) in enumerate(dataloader):
        optimizer.zero_grad()

        predictions = model(
            pv_features.to(device, dtype=torch.float),
            #hrv_features.to(device, dtype=torch.float),
            nwp_features.to(device, dtype=torch.float),
        )

        loss = criterion(predictions, pv_targets.to(device, dtype=torch.float))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad_norm)
        optimizer.step()

        size = int(pv_targets.size(0))
        running_loss += float(loss) * size
        count += size

        if i % 100 == 3: 
            print(f"Epoch {epoch + 1}, {i + 1}: loss: {running_loss / count}, time: {time[0]}") 
            os.makedirs("submission", exist_ok=True)
            torch.save(model.state_dict(), "submission/model_weather_only.pt")

            #sample_pv, sample_vis = util.visualize_example(
                #pv_features[0], pv_targets[0], predictions[0], nwp_features[0]
            #)

            if i % 500 == 3:
                st = datetime.now()
                print(f"validating: start {datetime.now()}")
                validation_loss = eval(eval_loader, model)
                print(f"loss: {validation_loss}, validation time {datetime.now() - st}")

            wandb.log({
                "train_loss": running_loss / count,
                "validation_loss": validation_loss,
                #"sample_pv": sample_pv,
                #"sample_vis": sample_vis,
            })

    print(f"Epoch {epoch + 1}: {running_loss / count}")

# Save your model
os.makedirs("submission", exist_ok=True)
torch.save(model.state_dict(), "submission/model_weather_only.pt")

wandb.finish()
