from datetime import datetime, time, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
from data.data import ChallengeDataset
from submission.model import Model
#from submission.resnet import Model
from submission.config import config
from util import util

def eval(dataloader, model, criterion=nn.L1Loss()):
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tot_loss, count = 0, 0

    model.to(device)


    for i, data in enumerate(dataloader):
        data = [dat.to(device, dtype=torch.float) for dat in data]
        pv_targets = data[3]
        data = [data[2], data[4], data[5]]
        predictions = model(
            *data
        )

        loss = criterion(predictions, pv_targets.to(device, dtype=torch.float))

        size = int(pv_targets.size(0))
        tot_loss += float(loss) * size
        count += size

    model.train()
    #del loss, data
    #torch.cuda.empty_cache()

    return (tot_loss / count)

if __name__ == "__main__":
    model = Model()
    model.load_state_dict(torch.load("./submission/model.pt"))
    print("model loaded")
    dataset = ChallengeDataset("nonhrv", 2020, eval=True, eval_year = 2021, eval_day=15)
    dataloader = DataLoader(dataset, batch_size=config.train.batch_size, pin_memory=True)
    print("starting eval")
    print("eval: ", eval(dataloader, model))
