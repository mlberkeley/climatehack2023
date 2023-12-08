import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from ocf_blosc2 import Blosc2
from torchinfo import summary
import json

from data import create_dataset
from submission.model import Model

plt.rcParams["figure.figsize"] = (20, 12)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pre-computed indices corresponding to each solar PV site stored in indices.json
with open("indices.json") as f:
    site_locations = {
        data_source: {
            int(site): (int(location[0]), int(location[1]))
            for site, location in locations.items()
        } for data_source, locations in json.load(f).items()
    }

dataloader = create_dataset("hrv", 2021, site_locations, 32)

summary(Model(), input_size=[(1, 12), (1, 12, 128, 128)])

# Actually do the training wow
model = Model().to(device)
criterion = nn.MSELoss()
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
