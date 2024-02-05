import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import config
from util import util


def eval(dataloader, model, criterion=nn.L1Loss()):
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tot_loss_1h, tot_loss_4h, count = 0, 0, 0

    with torch.no_grad():
        for i, (pv_features, meta, nonhrv, weather, pv_targets) in enumerate(dataloader):
            meta, nonhrv, weather = util.dict_to_device(meta), util.dict_to_device(nonhrv), util.dict_to_device(weather)
            pv_features = pv_features.to(device, dtype=torch.float)
            pv_targets = pv_targets.to(device, dtype=torch.float)

            predictions = model(pv_features, meta, nonhrv, weather)

            loss_1h = criterion(predictions[:, :12], pv_targets[:, :12])
            loss_4h = criterion(predictions, pv_targets)

            size = int(pv_targets.size(0))
            tot_loss_1h += float(loss_1h) * size
            tot_loss_4h += float(loss_4h) * size
            count += size

    model.train()

    return tot_loss_1h / count, tot_loss_4h / count


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
