import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

import h5py
import torch

from competition import BaseEvaluator
from resnet import MainModel as Model
import numpy as np
from util import util


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluator(BaseEvaluator):
    def setup(self) -> None:
        """Sets up anything required for evaluation, e.g. loading a model."""

        self.model = Model().to(device)
        self.model.load_state_dict(torch.load("pushtest.pt", map_location=device))
        self.model.eval()

    def predict(self, features: h5py.File):
        """Makes solar PV predictions for a test set.

        You will have to modify this method in order to use additional test set data variables
        with your model.

        Args:
            features (h5py.File): Solar PV, satellite imagery, weather forecast and air quality forecast features.

        Yields:
            Generator[np.ndarray, Any, None]: A batch of predictions.
        """

        with torch.inference_mode():
            #for pv, hrv, weather in self.batch(features, variables=["pv", "nonhrv", "weather"], batch_size=32):
            #for data_tuple in self.batch(features, variables=["pv", "alb_rad", "aswdifd_s", "aswdir_s", "cape_con", "clch", "clcl", "clcm", "clct", "h_snow", "omega_1000", "omega_700", "omega_850", "omega_950", "pmsl", "pv", "relhum_2m", "runoff_g", "runoff_s", "t_2m", "t_500", "t_850", "t_950", "t_g", "td_2m", "tot_prec", "u_10m", "u_50", "u_500", "u_850", "u_950", "v_10m", "v_50", "v_500", "v_850", "v_950", "vmax_10m", "w_snow"], batch_size=32):
            for data_tuple in self.batch(features, variables=["pv", "nonhrv", "clch", "clcl", "clcm", "clct", "latitude", "longitude", "orientation", "tilt", "kwp"], batch_size=32): 
#"clch", "clcl", "clcm", "clct", "h_snow", "w_snow", "t_g", "t_2m", "tot_prec"], batch_size=32):
                # Produce solar PV predictions for this batch
                nonhrv = data_tuple[1]
                nonhrv = np.stack([nonhrv[..., channel] for channel in [1, 7, 8]], axis = 1)
                pv = data_tuple[0]
                nwp = np.stack([x for x in data_tuple[2:6]], axis=1)

                #nwp = torch.concat([torch.from_numpy(x) for x in data_tuple[2:]], dim=1)
                site_features = np.stack([x for x in data_tuple[-5:]], axis = -1)
                site_features = util.site_normalize(torch.from_numpy(site_features).to(device))

                yield self.model(
                    torch.from_numpy(pv).to(device),
                    site_features,
                    torch.from_numpy(nonhrv).to(device),
                    torch.from_numpy(nwp).to(device),

                ).to("cpu")


if __name__ == "__main__":
    Evaluator().evaluate()
