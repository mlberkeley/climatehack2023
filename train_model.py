import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchinfo import summary
import json

from data import ChallengeDataset
from torch.utils.data import DataLoader
from submission.model import Model

from submission.config import config

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == "cpu":
    print("====================")
    for i in range(10):
        print("YOU ARE IN CPU MODE")


summary(Model(), input_size=[(1, 12), (1, 12, 128, 128)])
data = "nonhrv"
year = 2020

# Actually do the training wow
model = Model().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.train.lr)

dataset = ChallengeDataset(data, year)
dataloader = DataLoader(dataset, batch_size=config.train.batch_size, pin_memory=True)

for epoch in range(config.train.num_epochs):
    model.train()

    running_loss = 0.0
    count = 0
    for i, (time, site, pv_features, hrv_features, pv_targets) in enumerate(dataloader):
        optimizer.zero_grad()

        predictions = model(
            pv_features.to(device, dtype=torch.float),
            hrv_features.to(device, dtype=torch.float),
        )

        loss = criterion(predictions, pv_targets.to(device, dtype=torch.float))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad_norm)
        optimizer.step()

        size = int(pv_targets.size(0))
        running_loss += float(loss) * size
        count += size

        if i % 200 == 199:
            print(f"Epoch {epoch + 1}, {i + 1}: loss: {running_loss / (count + .0000001)}")
            os.makedirs("submission", exist_ok=True)
            torch.save(model.state_dict(), "submission/model.pt")

    print(f"Epoch {epoch + 1}: {running_loss / (count + .0000001)}")

# Save your model
os.makedirs("submission", exist_ok=True)
torch.save(model.state_dict(), "submission/model.pt")
