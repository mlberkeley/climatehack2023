# ClimateHack 2023
This repository contains the project files for ML@B's ClimateHack 2023 contest submissions.

##  Training a Model
Build the appropriate model, then import it in `train_model.py`. Modify appropriate configs/hyperparams in `submission/config.py`. Run
```
python train_model.py -n "run_name" -m "run_notes"
```
to begin training and log in wandb under `run_name`. Run without flags to use defaults. 

## Submission
Before submitting, be sure to check `submission/run.py` to make sure that the `vars` array on line 13 is consistent with the variables you trained your model on. Then, run
```
bash submit.sh
```
to login to Doxa and submit your model.
