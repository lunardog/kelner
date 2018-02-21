from __future__ import absolute_import

import click

from . import kelner_model
from ..utils import get_file


class TensorflowModel(kelner_model.KelnerModel):

    def __init__(
            self,
            model_file_name,
            input_node_name=None,
            output_node_name=None,
            flags=[]
    ):
        import tensorflow as tf
        self.session = tf.Session()
        self.input_node_name = input_node_name
        self.output_node_name = output_node_name
        file_name = get_file(
            model_file_name,
            extract=('EXTRACT' in flags)
        )
        with self.session.graph.as_default():
            with tf.gfile.FastGFile(file_name, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
        if input_node_name is not None:
            self.set_input(input_node_name)
        if output_node_name is not None:
            self.set_output(output_node_name)

    def __call__(self, data):
        """
        Run inference on a batch of data
        """
        with self.session.as_default():
            prediction = self.session.run(self.output,
                                          {self.input_node_name + ':0': data})
            return prediction

    def get_op(self, op_name):
        """
        Get graph operation by name
        """
        graph = self.session.graph
        return graph.get_tensor_by_name(op_name + ':0')

    def set_input(self, input_node_name):
        """
        Set the input operation to be used for inference
        """
        self.input_node_name = input_node_name
        self.input = self.get_op(self.input_node_name)

    def set_output(self, output_node_name):
        """
        Set the output operation to be used for inference
        """
        self.output_node_name = output_node_name
        self.output = self.get_op(self.output_node_name)

    def summary(self):
        """
        Print a model summary
        """
        edges = []
        summary = ''
        with self.session.graph.as_default():
            graph_def = self.session.graph.as_graph_def()
            for node in graph_def.node:
                if node.input is not None:
                    for inp in node.input:
                        edges.append((inp, node.name, node.op))
            summary += 'digraph g\n'
            summary += '{\n  node [shape=plaintext];\n  \n'
            for left, right, op in edges:
                summary += '  %s -> %s [label=%s];\n' % (left, right, op)
            summary += '}\n'
        click.echo(summary)


def load(
        model_name,
        input_node_name=None,
        output_node_name=None,
        flags=[]
):
    click.echo(
        'Loading a TensorFlow model from %s...' % (model_name),
        err=True
    )
    model = TensorflowModel(
        model_name, input_node_name, output_node_name, flags
    )
    return model
