## ClimateHack 2023
This repository contains the project files for ML@B's ClimateHack 2023 contest submissions.

###  Training a Model
Specify the model name in `config.model.name` with the model name in `build.py`. We use a `yaml` file for each model, for easy access.

```
python main.py -n run_name -c config_filepath
```
Specify `run_name` to for wandb logging. Run without flags to use defaults. Run name and configs must be specified.

The model weights and a json copy of the config file used will be saved in `ckpts/{run_name}/`.

### Local Evaluation
Local eval:
```
python main.py -n run_name -c config_filepath -t eval
```
(default behaviour is that `main.py` will train, not eval)

DOXA local eval:
```
python doxa_local.py ckpts/run_name
```
This automatically copies the model weights and config from the folder `ckpts/{run_name}/` to the submissions folder, then runs the eval on the model. We recommend running this before submission to make sure everything works as intended!

### Submission
```
bash submit.sh ckpts/run_name
```
Logs into DOXA and submits model.
