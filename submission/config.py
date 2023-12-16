from easydict import EasyDict as edict
from datetime import datetime

config = edict()

# TRAIN CONFIG
config.train = edict()
config.train.batch_size = 32
config.train.num_epochs = 1
config.train.num_workers = 8
config.train.lr = 1e-3
config.train.clip_grad_norm = 1.

# DATA CONFIG
config.data = edict()
config.data.hrv_path = "/data/climatehack/official_dataset/hrv"
config.data.nonhrv_path = "/data/climatehack/official_dataset/nonhrv"
config.data.start_date = datetime(2020, 7, 1)
config.data.end_date = datetime(2020, 7, 30)
config.data.channel = 8
