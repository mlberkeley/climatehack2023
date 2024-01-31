import numpy as np
import torch
import keys

normalizers = np.array([(-1.4087231368096667, 1.3890225067478146),
 (52.978670018652174, 1.4779275085314911),
 (31.545373449030897, 6.077633030692922),
 (3.017692725650799, 3.460296728623524),
 (178.186237936907, 47.040837795130464)])
device = "cuda" if torch.cuda.is_available() else "cpu"
means, stds = torch.tensor(
        normalizers[:, 0],
        dtype=torch.float, device = device
    ), torch.tensor(
        normalizers[:, 1],
        dtype=torch.float, device = device
    )

def site_normalize(vals):
    if keys.META.LATITUDE in vals:
        vals[keys.META.LATITUDE] = (vals[keys.META.LATITUDE] - means[0]) / stds[0]
    if keys.META.LONGITUDE in vals:
        vals[keys.META.LONGITUDE] = (vals[keys.META.LONGITUDE] - means[1]) / stds[1]
    if keys.META.ORIENTATION in vals:
        vals[keys.META.ORIENTATION] = (vals[keys.META.ORIENTATION] - means[2]) / stds[2]
    if keys.META.TILT in vals:
        vals[keys.META.TILT] = (vals[keys.META.TILT] - means[3]) / stds[3]
    if keys.META.KWP in vals:
        vals[keys.META.KWP] = (vals[keys.META.KWP] - means[4]) / stds[4]
    # return (vals - means) / stds
    return vals

#def denormalize(val_type, val):
    #assert val_type in norms, "val_type error, not in norms, can't be normalized"
    #mean, std = norms[val_type]
    #return (val * std) + mean
