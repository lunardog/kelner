from .models import kelner_model, KerasModel, TensorflowModel


def load(model_file_name, engine="keras", *args, **kwargs):
    """ Loads a saved model, given engine """
    if engine == "keras":
        m = KerasModel(model_file_name, *args, **kwargs)
    elif engine == "tensorflow" or engine == "tf":
        m = TensorflowModel(model_file_name, *args, **kwargs)
    else:
        raise ValueError("The engine param must be one of [keras, tensorflow, tf]")
    return m
