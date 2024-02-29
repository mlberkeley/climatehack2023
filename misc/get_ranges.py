from tqdm import tqdm
import json
import os

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from ocf_blosc2 import Blosc2


for source in (
        'AEROSOLS',
        'WEATHER',
    ):
    xrdata = xr.open_mfdataset(
        f'/data/climatehack/official_dataset/{source.lower()}/*/*.zarr.zip',
        engine="zarr",
        consolidated=True,
    )

    for i, key in enumerate(xrdata.data_vars):
        a = xrdata[key].to_numpy()

        # plt.hist(a.flatten(), bins=100, histtype='step')
        # plt.title(f'{source}.{key}')
        # plt.savefig(f'ranges/{source}.{key}.png')
        # plt.close()

        print(f'{source}.{key} = mean, std: ({np.nanmean(a)}, {np.nanstd(a)})')
        print(f'{source}.{key} = min,  max: ({np.nanmin(a)}, {np.nanmax(a)})')
        print(f'{source}.{key} = shape: {a.shape}, {a.dtype}')
        print(f'{source}.{key} = nan count: {np.isnan(a).sum()} / {a.size}')
