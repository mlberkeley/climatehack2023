# $1 = path to the model folder, should be something like
#       ckpts/model_name
# maybe cp the config as well and then mparse the config in run.py to figure out what model to import
cp $1/best_ema.pt submission/model.pt

python3 -m doxa_cli login
python3 -m doxa_cli upload submission
