from __future__ import absolute_import

import keras
import click

from . import kelner_model
from ..utils import get_file


class KerasModel(kelner_model.KelnerModel):

    def __init__(
            self,
            model_file_name,
            input_node_name=None,
            output_node_name=None,
            flags=[]
    ):
        custom_objs = {}
        if 'USE_MOBILENET' in flags:
            custom_objs = {
                'relu6': keras.applications.mobilenet.relu6,
                'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D
            }
        file_name = get_file(
            model_file_name,
            extract=('EXTRACT' in flags)
        )
        model = keras.models.load_model(
            file_name,
            compile=False,
            custom_objects=custom_objs
        )
        self._set_model(model)

    def __call__(self,  data):
        """
        Infers on batch
        """
        return self.model.predict(data)

    def _set_model(self, model):
        self.model = model
        if model is not None:
            self.input = self.model.inputs[0]
            self.output = self.model.outputs[0]
        else:
            self.input, self.output = None, None

    def summary(self):
        return self.model.summary()


def load(
        model_name,
        input_node_name=None,
        output_node_name=None,
        flags=[]
):
    click.echo('Loading a Keras model from %s...' % (model_name),
               err=True)
    model_cls = KerasModel

    model = model_cls(model_name, input_node_name, output_node_name, flags)
    return model
