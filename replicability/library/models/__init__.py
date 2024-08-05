from library.models.bigru import BiGRU
from library.models.cnn import CNN
from library.models.transformer import Transformer


def get_predictor(model_type: str, hps: dict):
    if model_type == "cnn":
        return CNN(**hps)
    if model_type == "transformer":
        return Transformer(**hps)
    if model_type == "gru":
        return BiGRU(**hps)

    raise ValueError(f"Unknown model type: {model_type}")
