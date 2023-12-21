import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))


import h5py
import torch
from competition import BaseEvaluator
from resnet import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluator(BaseEvaluator):
    def setup(self) -> None:
        """Sets up anything required for evaluation, e.g. loading a model."""

        self.model = Model().to(device)
        self.model.load_state_dict(torch.load("model_random.pt", map_location=device))
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
            # Select the variables you wish to use here!
            #for pv, hrv, weather in self.batch(features, variables=["pv", "nonhrv", "weather"], batch_size=32):
            for pv, hrv in self.batch(features, variables=["pv", "nonhrv"], batch_size=32):
                # Produce solar PV predictions for this batch
                hrv = hrv[...,8]
                #nwp = weather[...]
                a = self.model(
                    torch.from_numpy(pv).to(device),
                    torch.from_numpy(hrv).to(device),
                    #torch.from_numpy(nwp).to(device),
                )
                yield a


if __name__ == "__main__":
    Evaluator().evaluate()
