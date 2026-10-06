                       


from importlib import import_module
from typing import Any

from models import SUPPORTED_MODEL_TYPES


_MODEL_MODULES = {
    "NTN": "models.ntn",
    "GIN": "models.gin",
    "GAT": "models.gat",
    "GCN": "models.gcn",
}


def normalize_model_type(model_type: Any) -> str:

    if model_type is None:
        model_type = "NTN"
    value = str(model_type).strip().upper()
    aliases = {
        "HIGNN": "NTN",
        "HIGIN": "GIN",
        "HIGAT": "GAT",
        "HIGCN": "GCN",
    }
    value = aliases.get(value, value)
    if value not in _MODEL_MODULES:
        supported = ", ".join(SUPPORTED_MODEL_TYPES)
        raise ValueError(f"Unsupported MODEL.TYPE={model_type!r}. Choose one of: {supported}")
    return value


def get_model_module(model_type: Any):

    model_type = normalize_model_type(model_type)
    return import_module(_MODEL_MODULES[model_type])


def get_model_type_from_cfg(cfg) -> str:

    model_type = getattr(getattr(cfg, "MODEL", None), "TYPE", "NTN")
    return normalize_model_type(model_type)


def build_model(cfg):

    model_type = get_model_type_from_cfg(cfg)
    module = get_model_module(model_type)
    model = module.build_model(cfg)
                                                                                
                                        
    model.model_type = model_type
    return model


def _module_for_model(model):

    model_type = getattr(model, "model_type", None)
    if model_type is not None:
        return get_model_module(model_type)

    module_name = model.__class__.__module__
    if module_name.startswith("models."):
        return import_module(module_name)

    raise ValueError(
        "Cannot determine model implementation module. Build the model with model.build_model(cfg)."
    )


def extract_attention_weights_from_model(model):
    return _module_for_model(model).extract_attention_weights_from_model(model)


def register_attention_hooks(model):
    return _module_for_model(model).register_attention_hooks(model)


def remove_attention_hooks(hooks, model=None):
                                                                          
    if model is not None:
        return _module_for_model(model).remove_attention_hooks(hooks)
    for hook in hooks:
        hook.remove()


def extract_model_attention(model, data):
    return _module_for_model(model).extract_model_attention(model, data)


__all__ = [
    "SUPPORTED_MODEL_TYPES",
    "normalize_model_type",
    "get_model_module",
    "get_model_type_from_cfg",
    "build_model",
    "extract_attention_weights_from_model",
    "register_attention_hooks",
    "remove_attention_hooks",
    "extract_model_attention",
]
