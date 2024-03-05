from .models import ResNetPV, MainModel2
from .vit import VIT_Model

# Note that config is now a regular dictionary, NOT an easydict
def build_model(config):
    model_type = config['model']['name']
    model_config = config['model']['config']

    if model_type == 'resnetpv':
        model = ResNetPV(model_config)
    elif model_type == 'mainmodel2':
        model = MainModel2(model_config)
    elif model_type == 'vit':
        model = VIT_Model(model_config)
    else:
        raise NotImplementedError(f"Unknown model: {model_type}")

    return model

