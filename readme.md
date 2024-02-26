# ClimateHack 2023
This repository contains the project files for ML@B's ClimateHack 2023 contest submissions.

##  Training a Model
First, make sure that you have an appropriate config file with the name of the model specified in `config.model.name`. Make sure this name corresponds with the model in `build.py`. **I would highly recommend creating a yaml file for each model.**

Then, run
```
python main.py -n run_name -c config_filepath
```
to begin training and log in wandb under `run_name`. Run without flags to use defaults. Run name and configs must be specified.

The model weights and a json copy of the config file used will be saved in `ckpts/{run_name}/`.

## Local Evaluation
To locally evaluate your model, run
```
python main.py -n run_name -c config_filepath -t eval
```
By default, `main.py` will train, not eval.

Another method of local evaluation that more closely simulates the Doxa platform is to run
```
python doxa_local.py run_name
```
This automatically copies the model weights and config from the folder `ckpts/{run_name}/` to the submissions folder, then runs the eval on the model. I would recommend running this before submission to make sure everything works as intended.

## Submission
Run
```
bash submit.sh run_name
```
to login to Doxa and submit your model.
