import boto3
from moto import mock_s3
import os

from kelner.models import keras_model
from kelner import utils

from .fixtures import runner
from .fixtures import sample_model
from .fixtures import sample_mobilenet_model
from .fixtures import sample_labels
from .fixtures import named_temp_file


def test_load_keras_model(runner, sample_model, named_temp_file):
    # save a model and try loading it
    model_file_name = named_temp_file.name
    sample_model.save(model_file_name)
    model = keras_model.load(model_file_name, flags=[])
    assert model is not None
    named_temp_file.close()


def test_load_mobilenet_model(runner, sample_mobilenet_model, named_temp_file):
    # save a model and try loading it
    model_file_name = named_temp_file.name
    sample_mobilenet_model.save(model_file_name)
    model = keras_model.load(model_file_name, flags=['USE_MOBILENET'])
    assert model is not None
    named_temp_file.close()


def test_load_remote_model(runner, sample_model, named_temp_file):
    bucket_name = 'test_bucket'
    model_file_name = named_temp_file.name
    region_name = 'us-east-1'
    key_name = os.path.basename(model_file_name)
    sample_model.save(model_file_name)

    mock = mock_s3()
    mock.start()

    conn = boto3.client('s3', region_name=region_name)
    assert conn is not None
    try:
        result = conn.create_bucket(Bucket=bucket_name)
        assert result

        conn.upload_file(model_file_name, bucket_name, key_name)
        remote_url = 's3://{}/{}'.format(
            bucket_name, key_name
        )
        model = keras_model.load(remote_url, flags=[])
        assert model is not None
        for index, layer in enumerate(model.model.layers):
            assert layer.name == sample_model.layers[index].name
    finally:
        named_temp_file.close()
        conn.delete_object(Bucket=bucket_name, Key=key_name)
        conn.delete_bucket(Bucket=bucket_name)
        mock.stop()
