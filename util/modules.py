from .sunincidence import siteinfo2projectdata, getSolarPosition
from datetime import datetime
from submission.resnet import *

def solar_pos(site_features, device):
    meta_keys = [
        keys.META.TIME,
        keys.META.LATITUDE,
        keys.META.LONGITUDE,
        keys.META.ORIENTATION,
        keys.META.TILT,
    ]
    
    meta = torch.stack([site_features[key] for key in meta_keys], dim=1)
    sun_pos = np.zeros((meta.shape[0], 2))

    for i, batch in enumerate(meta):
        if not batch[0]:
            return torch.Tensor([[0, 0]]).to(device)

        proj = siteinfo2projectdata(batch[1].cpu(), batch[2].cpu(), batch[3].cpu(), batch[4].cpu())

        timestamp_dt = datetime.utcfromtimestamp(batch[0].cpu().item())
        zenith, incident = getSolarPosition(t=timestamp_dt, project_data=proj)

        sun_pos[i, 0] = zenith
        sun_pos[i, 1] = incident
    
    return torch.from_numpy(sun_pos).float().to(device)