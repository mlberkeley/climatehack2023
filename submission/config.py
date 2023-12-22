from easydict import EasyDict as edict
from datetime import datetime

config = edict()

# TRAIN CONFIG
config.train = edict()
config.train.batch_size = 128
config.train.num_epochs = 500
config.train.num_workers = 8
config.train.lr = 1e-3
config.train.clip_grad_norm = 1
config.train.random_site_threshold = .02
config.train.random_time_threshold = .5
#config.train.weather_keys = ['alb_rad', 'clch', 'clcl', 'clcm', 'clcl', 'clct', 'h_snow', 't_2m']
#config.train.weather_keys = ["alb_rad", "aswdifd_s", "aswdir_s", "cape_con", "clch", "clcl", "clcm", "clct", "h_snow", "omega_1000", "omega_700", "omega_850", "omega_950", "pmsl", "relhum_2m", "runoff_g", "runoff_s", "t_2m", "t_500", "t_850", "t_950", "t_g", "td_2m", "tot_prec", "u_10m", "u_50", "u_500", "u_850", "u_950", "v_10m", "v_50", "v_500", "v_850", "v_950", "vmax_10m", "w_snow"]
#config.train.weather_keys = ["alb_rad", "aswdir_s", "cape_con", "clch", "clcl", "clcm", "clct", "h_snow", "t_g", "tot_prec", "v_500", "v_850", "w_snow"]
config.train.weather_keys = ["clch", "clcl", "clcm", "clct", "h_snow", "w_snow", "t_g", "t_2m", "tot_prec"]
#config.train.weather_keys = ["clch", "clcl", "clcm", "clct"]
config.train.weather_groups = [4, 2, 2, 1]
#config.train.weather_groups = [4]

# DATA CONFIG
config.data = edict()
config.data.hrv_path = "/data/climatehack/official_dataset/hrv"
config.data.nonhrv_path = "/data/climatehack/official_dataset/nonhrv"
config.data.start_date = datetime(2020, 1, 1)
config.data.end_date = datetime(2020, 12, 30)
config.data.channel = 8
