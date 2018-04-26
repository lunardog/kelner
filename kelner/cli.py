import click

from . import server
from . import client
from . import imagenet

CONTEXT_SETTINGS = dict(
    default_map={
        'model': None,
    }
)


def load_labels(label_file):
    lines = label_file.readlines()
    the_labels = [line.strip() for line in lines]
    the_labels = [label for label in the_labels if len(label) > 0]
    return the_labels


@click.group()
@click.option(
    '--url', '-u',
    default='http://localhost:%d' % (server.KELNER_PORT),
    help='Kelner server url',
    envvar='KELNER_URL'
)
@click.version_option()
@click.pass_context
def kelner(ctx, url):
    ctx.obj = client.KelnerClient(url)
    pass


@kelner.command()
@click.argument(
    'file',
    type=click.Path(exists=True),
    nargs=1
)
@click.pass_context
def infer(ctx, file):
    me = ctx.obj
    contents, mimetype, size = client.decode_file(file)
    response = me.request(contents, mimetype, size)
    click.echo(response)


@kelner.command()
@click.argument(
    'file',
    type=click.Path(exists=True),
    nargs=1
)
@click.option(
    '--labels',
    type=str,
    default='',
    help='Comma-separated list of labels'
)
@click.option(
    '--labels-file',
    type=click.File('r'),
    help='Path to a file with labels, one per line'
)
@click.option(
    '--imagenet-labels',
    is_flag=True
)
@click.option(
    '--top',
    type=int,
    default=1
)
@click.pass_context
def classify(ctx, file, labels, labels_file, imagenet_labels, top):
    labels = labels.split(',')
    if labels_file is not None:
        labels = labels_file.readlines()
    if imagenet_labels:
        labels = imagenet.IMAGENET_LABELS
    me = ctx.obj
    scores = me.classify(file, labels=labels, top=top)
    for label, score in scores:
        click.echo("%s: %f" % (label, score))


@click.group(invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option(
    '--load-model', '-m',
    help='Load a model',
    required=True,
    envvar='KELNER_MODEL')
@click.option(
    '--extract',
    help='Extract the model file',
    is_flag=True)
@click.option(
    '--engine',
    type=click.Choice(['keras', 'tensorflow']),
    default='keras',
    envvar='KELNER_ENGINE',
    help='Engine to load and run the model')
@click.option(
    '--input-node',
    default=None,
    envvar='KELNER_MODEL_INPUT',
    help='Name of the input node')
@click.option(
    '--output-node',
    default=None,
    envvar='KELNER_MODEL_OUTPUT',
    help='Name of the output node')
@click.option(
    '--host', '-h',
    default='0.0.0.0',
    help='Address to listen on')
@click.option(
    '--port', '-p',
    default=server.KELNER_PORT,
    help='Port to listen on (default %d)' % (server.KELNER_PORT))
@click.option(
    '--use-mobilenet',
    is_flag=True)
@click.option('--dry-run', is_flag=True)
@click.version_option()
def kelnerd(
        ctx,
        load_model,
        extract,
        engine,
        input_node,
        output_node,
        host,
        port,
        use_mobilenet,
        dry_run
):
    """ Serves Keras and Tensorflow models """
    ctx.obj = {}
    from . import models

    flags = []
    if use_mobilenet:
        flags += ['USE_MOBILENET']
    if extract:
        flags += ['EXTRACT']

    if engine == 'keras':
        loaded_model = models.keras_model.load(
            load_model, input_node, output_node, flags=flags
        )
    else:
        loaded_model = models.tensorflow_model.load(
            load_model, input_node, output_node, flags=flags
        )

    ctx.obj['model'] = loaded_model

    if ctx.invoked_subcommand is None:

        if engine == 'tensorflow':
            if input_node is None or output_node is None:
                raise click.BadOptionUsage('Serving Tensorflow models' +
                                           ' required input and output nodes' +
                                           ' to be specified', ctx)

        try:
            k_server = server.KelnerServer(loaded_model)
            click.echo('Listening on %s:%d' % (host, port), err=True)
            if not dry_run:
                k_server.serve_http(host, port)

        except OSError as e:
            click.echo(str(e), err=True)


@kelnerd.command()
@click.pass_context
def info(ctx):
    """ Prints model information """
    try:
        ctx.obj['model'].summary()
    except Exception as e:
        click.echo(str(e), err=True)
        pass
