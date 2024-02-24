from models import ResNetPV, NonHRVMeta, MetaAndPv, MetaAndPv5, MainModel2
from vit import VIT_Model

def build_model(config):
    "Model builder."

    model_type = config.model.name

    if model_type == 'resnetpv':
        model = ResNetPV(config.model.config)
    elif model_type == 'nonhrvmeta':
        model = NonHRVMeta(config.model.config)
    elif model_type == 'metaandpv':
        model = MetaAndPv(config.model.config)
    elif model_type == 'metaandpv5':
        model = MetaAndPv5(config.model.config)
    elif model_type == 'mainmodel2':
        model = MainModel2(config.model.config)
    elif model_type == 'vit':
        model = VIT_Model(config.model.config)
    else:
        raise NotImplementedError(f"Unknown model: {model_type}")
    
    return model
 
