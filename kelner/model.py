from .models import kelner_model, KerasModel, TensorflowModel


def load(model_file_name, engine="keras", *args, **kwargs):
    m = None
    if engine == "keras":
        m = KerasModel(model_file_name, *args, **kwargs)
    elif engine == "tensorflow" or engine == "tf":
        m = TensorflowModel(model_file_name, *args, **kwargs)
    return m
