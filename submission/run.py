import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

import h5py
import torch

from competition import BaseEvaluator
from submission.models import MainModel2 as Model
import numpy as np
import util as util
import keys as keys


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluator(BaseEvaluator):
    def setup(self) -> None:
        """Sets up anything required for evaluation, e.g. loading a model."""

        self.model = Model(dict(channel='VIS008')).to(device)
        self.model.load_state_dict(torch.load("here comes the sun.pt.best_ema", map_location=device))
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
            for data_batch in self.batch(
                    features,
                    variables=[
                        'pv', 'nonhrv',
                    ] + [e.name.lower() for e in keys.META]
                      + [e.name.lower() for e in self.model.REQUIRED_WEATHER],
                      # + [e.name.lower() for e in keys.AEROSOLS],
                    batch_size=32
                ):
                # Produce solar PV predictions for this batch
                pv = torch.from_numpy(data_batch['pv']).to(device)
                nonhrv = data_batch['nonhrv']
                meta_features = {
                        k : torch.from_numpy(data_batch[k.name.lower()]).to(device)
                        for k in keys.META
                }
                meta_features[keys.META.TIME] = meta_features[keys.META.TIME] / 1e9 # nanos to seconds
                nonhrv_features = {
                        k: torch.from_numpy(nonhrv[..., k].transpose(0, 1, 3, 2)).to(device)
                        for k in self.model.REQUIRED_NONHRV
                }
                weather_features = {
                        k : torch.from_numpy(data_batch[k.name.lower()]).to(device)
                        for k in self.model.REQUIRED_WEATHER
                }
                # aerosol_features = {
                #         k : torch.from_numpy(data_batch[k.name.lower()]).to(device)
                #         for k in keys.AEROSOLS
                # }

                # INFO  this is ugly; but that's okay
                for k in self.model.REQUIRED_WEATHER:
                    weather_features[k] = (weather_features[k] - keys.WEATHER_RANGES[k][0]) / (keys.WEATHER_RANGES[k][1] - keys.WEATHER_RANGES[k][0])
                # TODO  normalize aerosol features
                # this normalization is currently done in model
                # needs to stay in model so that we don't have to do it here.. perhaps?
                # site_features = util.site_normalize(torch.from_numpy(site_features).to(device))

                yield self.model(pv, meta_features, nonhrv_features, weather_features).cpu()


if __name__ == "__main__":
    Evaluator().evaluate()
