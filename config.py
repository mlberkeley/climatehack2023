from easydict import EasyDict as edict
from datetime import datetime

config = edict()


# TRAIN CONFIG
config.train = edict()
config.train.model_save_name = "pushtest.pt"
config.train.batch_size = 256
config.train.num_epochs = 1000
config.train.lr = 1e-3
config.train.clip_grad_norm = 1
config.train.log_every = 10
config.train.eval_every = 250


# EVAL CONFIG
config.eval = edict()
config.eval.batch_size = 256


# DATA CONFIG
config.data = edict()

config.data.num_workers = 16
config.data.root = "/data/climatehack/"

config.data.train_start_date = datetime(2021, 1, 1)
config.data.train_end_date = datetime(2022, 1, 1)  # end date not inclusive
# subsets are randomly sampled from the full dataset using a seed of 21
config.data.train_subset_size = 0  # 0 means use all data

config.data.eval_start_date = datetime(2020, 1, 1)
config.data.eval_end_date = datetime(2021, 1, 1)
config.data.eval_subset_size = 10000

