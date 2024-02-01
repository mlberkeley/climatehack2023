import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

import h5py
import torch

from competition import BaseEvaluator
from resnet import ResNet34 as Model
import numpy as np
from util import util
import keys as keys


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluator(BaseEvaluator):
    def setup(self) -> None:
        """Sets up anything required for evaluation, e.g. loading a model."""

        self.model = Model().to(device)
        self.model.load_state_dict(torch.load("pushtest.pt", map_location=device))
        # self.model = torch.load("pushtest.pt", map_location=device)
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
                        "pv", "nonhrv",
                    ] + [e.name.lower() for e in keys.META]
                      + [e.name.lower() for e in keys.WEATHER],
                    batch_size=32
                ):
                # Produce solar PV predictions for this batch
                pv = data_batch['pv']
                nonhrv = data_batch['nonhrv']
                a = {
                    e: torch.from_numpy(nonhrv[..., e.value]).to(device) for e in keys.NONHRV
                }

                site_features = {
                    k : data_batch[k.name.lower()] for k in keys.META
                }
                weather_features = {
                    k : data_batch[k.name.lower()] for k in keys.WEATHER
                }
                for k in keys.WEATHER:
                    weather_features[k] = (weather_features[k] - keys.WEATHER_RANGES[k][0]) / (keys.WEATHER_RANGES[k][1] - keys.WEATHER_RANGES[k][0])
                # normalization is currently done in model
                # needs to stay in model so that we don't have to do it here..
                # site_features = util.site_normalize(torch.from_numpy(site_features).to(device))

                yield self.model(
                    torch.from_numpy(pv).to(device),
                    {
                        k : torch.from_numpy(v).to(device) for k, v in site_features.items()
                    },
                    {
                        e: torch.from_numpy(nonhrv[..., e.value]).to(device) for e in keys.NONHRV
                    },
                    {
                        k: torch.from_numpy(v).to(device) for k, v in weather_features.items()
                    }
                ).to("cpu")


if __name__ == "__main__":
    Evaluator().evaluate()
