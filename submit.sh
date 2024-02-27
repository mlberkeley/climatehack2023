# $1 = path to the model folder, should be something like
#       ckpts/model_name
# maybe cp the config as well and then mparse the config in run.py to figure out what model to import
cp ckpts/$1/model.pt.best_ema submission/model.pt
cp ckpts/$1/config.json submission/config.json

python3 -m doxa_cli login
python3 -m doxa_cli upload submission
