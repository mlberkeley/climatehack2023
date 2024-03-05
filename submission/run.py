import pathlib
import sys
import json

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

import h5py
import torch

from competition import BaseEvaluator
import numpy as np
import util as util
from models.keys import META, COMPUTED, HRV, NONHRV, WEATHER, AEROSOLS, WEATHER_RANGES, AEROSOLS_RANGES
from models.build import build_model
from modules.solar import solar_pos

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluator(BaseEvaluator):

    def setup(self) -> None:
        """Sets up anything required for evaluation, e.g. loading a model."""

        config = json.load(open('config.json'))

        self.model = build_model(config).to(device)
        self.model.load_state_dict(torch.load(f"model.pt", map_location=device))
        self.model.eval()

        features = self.model.required_features
        self.meta_features = [k for k in META]
        self.computed_features = [k for k in features if COMPUTED.has(k)]
        self.hrv_features = [k for k in features if HRV.has(k)]
        self.nonhrv_features = [k for k in features if NONHRV.has(k)]
        self.weather_features = [k for k in features if WEATHER.has(k)]
        self.aerosols_features = [k for k in features if AEROSOLS.has(k)]
        self.require_future_nonhrv = COMPUTED.FUTURE_NONHRV in features

    def predict(self, features: h5py.File):
        """Makes solar PV predictions for a test set.

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
                    ] + [e.name.lower()
                         for e in self.meta_features + self.hrv_features +
                                self.weather_features + self.aerosols_features
                    ],
                    batch_size=32
                ):
                # Produce solar PV predictions for this batch
                pv = torch.from_numpy(data_batch['pv']).to(device)
                nonhrv = data_batch['nonhrv']

                input_data = dict()

                for k in META:
                    input_data[k] = torch.from_numpy(data_batch[k.name.lower()])
                input_data[META.TIME] = input_data[META.TIME] / 1e9 # nanos to seconds

                input_data[COMPUTED.SOLAR_ANGLES] = solar_pos(input_data, device, hourly=True).view(-1, 6, 2)

                input_data |= {
                        k: torch.from_numpy(data_batch[k.name.lower()])
                        for k in self.hrv_features
                }
                input_data |= {
                        k: torch.from_numpy(nonhrv[..., k])
                        for k in self.nonhrv_features
                }
                input_data |= {
                        k : torch.from_numpy(data_batch[k.name.lower()])
                        for k in self.weather_features
                }
                input_data |= {
                        k : torch.from_numpy(data_batch[k.name.lower()]).to(device)
                        for k in self.aerosols_features
                }

                for k in self.weather_features:
                    vmin, vmax = WEATHER_RANGES[k]
                    input_data[k] = (input_data[k] - vmin) / (vmax - vmin)

                for k in self.aerosols_features:
                    vmin, vmax = AEROSOLS_RANGES[k]
                    input_data[k] = (input_data[k] - vmin) / (vmax - vmin)

                # TODO  normalize aerosol features
                # this normalization is currently done in model
                # needs to stay in model so that we don't have to do it here.. perhaps?
                # site_features = util.site_normalize(torch.from_numpy(site_features).to(device))
                input_data = util.dict_to_device(input_data, device)

                yield self.model(pv, input_data).cpu()


if __name__ == "__main__":
    Evaluator().evaluate()
