import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr

from ocf_blosc2 import Blosc2
from torch.utils.data import DataLoader
from torchinfo import summary
import json
from pathlib import Path

from submission.model import Model
from challenge_dataset import ChallengeDataset

plt.rcParams["figure.figsize"] = (20, 12)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming data already downloaded... see TODO some other file for this

# Load pv data by concatenating all data in this folder
# Can modify as needed to load specific data
data_dir = Path("/data/climatehack/official_dataset/pv/2020")
pv = pd.concat(
    pd.read_parquet(parquet_file).drop("generation_wh", axis=1)
    for parquet_file in data_dir.glob('*.parquet')
)

# Once again, this is opening multiple datasets at once
# hrv = xr.open_dataset("data/satellite-hrv/2020/7.zarr.zip", engine="zarr", chunks="auto")
# opens a single dataset
hrv = xr.open_mfdataset("/data/climatehack/official_dataset/nonhrv/2020/*.zarr.zip", engine="zarr", chunks="auto")

# pre-computed indices corresponding to each solar PV site stored in indices.json
with open("indices.json") as f:
    site_locations = {
        data_source: {
            int(site): (int(location[0]), int(location[1]))
            for site, location in locations.items()
        }
        for data_source, locations in json.load(f).items()
    }

summary(Model(), input_size=[(1, 12), (1, 12, 128, 128)])

# Actually do the training wow
BATCH_SIZE = 32

dataset = ChallengeDataset(pv, hrv, site_locations=site_locations)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, pin_memory=True)

model = Model().to(device)
criterion = nn.L1Loss()
optimiser = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 1

for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0
    count = 0
    for i, (pv_features, hrv_features, pv_targets) in enumerate(dataloader):
        optimiser.zero_grad()

        predictions = model(
            pv_features.to(device, dtype=torch.float),
            hrv_features.to(device, dtype=torch.float),
        )

        loss = criterion(predictions, pv_targets.to(device, dtype=torch.float))
        loss.backward()

        optimiser.step()

        size = int(pv_targets.size(0))
        running_loss += float(loss) * size
        count += size

        if i % 200 == 199:
            print(f"Epoch {epoch + 1}, {i + 1}: {running_loss / count}")

    print(f"Epoch {epoch + 1}: {running_loss / count}")

# Save your model
torch.save(model.state_dict(), "submission/model.pt")
