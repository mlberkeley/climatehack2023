from easydict import EasyDict as edict
from datetime import datetime

config = edict()

# TRAIN CONFIG
config.train = edict()
config.train.model_save_name = "noimage.pt"
config.train.batch_size = 256
config.train.num_epochs = 1000
config.train.lr = 1e-3
config.train.clip_grad_norm = 1
config.train.random_site_threshold = .02
config.train.random_time_threshold = .05
config.train.data_sources = ["pv", "nonhrv", "weather"]
config.train.minutes_resolution = 60
config.train.eval_resolution = 60
#config.train.weather_keys = ["alb_rad", "aswdifd_s", "aswdir_s", "cape_con", "clch", "clcl", "clcm", "clct", "h_snow", "omega_1000", "omega_700", "omega_850", "omega_950", "pmsl", "relhum_2m", "runoff_g", "runoff_s", "t_2m", "t_500", "t_850", "t_950", "t_g", "td_2m", "tot_prec", "u_10m", "u_50", "u_500", "u_850", "u_950", "v_10m", "v_50", "v_500", "v_850", "v_950", "vmax_10m", "w_snow"]
#config.train.weather_keys = ["alb_rad", "aswdir_s", "cape_con", "clch", "clcl", "clcm", "clct", "h_snow", "t_g", "tot_prec", "v_500", "v_850", "w_snow"]
config.train.weather_keys = ["clch", "clcl", "clcm", "clct", "h_snow", "w_snow", "t_g", "t_2m", "tot_prec"]
config.train.weather_groups = [4, 2, 2, 1]

# EVAL CONFIG
config.eval = edict()
config.eval.batch_size = 256

# DATA CONFIG
config.data = edict()
config.data.channel = 8
config.data.hrv_path = "/data/climatehack/official_dataset/hrv"
config.data.nonhrv_path = "/data/climatehack/official_dataset/nonhrv"
config.data.num_workers = 16

config.data.train_start_date = datetime(2020, 1, 1)
config.data.train_end_date = datetime(2021, 1, 1)  # end date not inclusive
# subsets are randomly sampled from the full dataset using a seed of 21
config.data.train_subset_size = 0 # 0 means use all data

config.data.eval_start_date = datetime(2021, 1, 1)
config.data.eval_end_date = datetime(2022, 1, 1)
config.data.eval_subset_size = 2560
