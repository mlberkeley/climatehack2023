import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from submission.config import config


def eval(dataloader, model, criterion=nn.L1Loss()):
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tot_loss, count = 0, 0

    with torch.no_grad():
        for i, (pv_features, meta, nonhrv, weather, pv_targets) in enumerate(dataloader):
            # pv_features, nonhrv_features, nwp_features, pv_targets, site_features = pv_features.to(device, dtype=torch.float), nonhrv_features.to(device, dtype=torch.float), nwp_features.to(device, dtype=torch.float), pv_targets.to(device, dtype=torch.float), site_features.to(device, dtype=torch.float)
            meta = {k: v.to(device, dtype=torch.float) for k, v in meta.items()}
            nonhrv = {k: v.to(device, dtype=torch.float) for k, v in nonhrv.items()}
            weather = {k: v.to(device, dtype=torch.float) for k, v in weather.items()}
            pv_features = pv_features.to(device, dtype=torch.float)
            pv_targets = pv_targets.to(device, dtype=torch.float)

            predictions = model(
                # pv_features,
                # nonhrv_features,
                # site_features
                pv_features, meta, nonhrv, weather
            )

            loss = criterion(predictions, pv_targets)

            size = int(pv_targets.size(0))
            tot_loss += float(loss) * size
            count += size

    model.train()

    return (tot_loss / count)


# WARN: uses old dataset
if __name__ == "__main__":
    from submission.model import MainModel
    from data.data import ChallengeDataset
    model = Model()
    model.load_state_dict(torch.load("./submission/model.pt"))
    print("model loaded")
    dataset = ChallengeDataset("nonhrv", 2020, eval=True, eval_year = 2021, eval_day=15)
    dataloader = DataLoader(dataset, batch_size=config.train.batch_size, pin_memory=True)
    print("starting eval")
    print("eval: ", eval(dataloader, model))
