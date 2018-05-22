from __future__ import absolute_import

import pytest
import keras
from keras import layers
import tempfile
import io
from click.testing import CliRunner
import threading
import json
from kelner import client
from kelner import server

# import requests_mock

try:
    # python 3
    import http.server as http
except:
    # python 2
    import BaseHTTPServer as http

SERVER_PORT = 8001
ECHO_SERVER_PORT = 8002
DEFAULT_LABELS = ["a", "b", "c"]
DEFAULT_RESPONSE = [[0.0, 1.0, 0.0]]
SAMPLE_DATA = [1.0, 2.0, 3.0]


class EchoModel(object):

    input = None
    output = None
    labels = []

    def __call__(self, data):
        return data


@pytest.fixture
def sample_file():
    with tempfile.NamedTemporaryFile(
            suffix='.json',
            mode='w',
            delete=False
    ) as f:
        f.write(json.dumps(SAMPLE_DATA))
        f.close()
    return f.name


@pytest.fixture(scope='function')
def runner():
    return CliRunner()


@pytest.fixture(scope='function')
def sample_model():
    inp = layers.Input(shape=(32,), name='input-01')
    out = layers.Dense(16, activation='softmax', name='output-01')(inp)
    mod = keras.models.Model(inputs=inp, outputs=out)
    return mod


@pytest.fixture(scope='function')
def sample_mobilenet_model():
    inp = layers.Input(shape=(32,), name='input-01')
    out = layers.Dense(16,
                       activation=keras.applications.mobilenet.relu6,
                       name='output-01')(inp)
    mod = keras.models.Model(inputs=inp, outputs=out)
    return mod


@pytest.fixture(scope='function')
def named_temp_file():
    f = tempfile.NamedTemporaryFile()
    return f


@pytest.fixture
def sample_labels():
    orig_labels = ['label-01', 'label-02', 'label-03']
    f = io.StringIO(u'\n'.join(orig_labels))
    return f, orig_labels


@pytest.fixture(scope='function')
def sample_echo_server():

    model = EchoModel()
    with server.KelnerServer(model) as s:
        httpd_thread = threading.Thread(
            target=s.serve_http,
            args=('127.0.0.1', ECHO_SERVER_PORT)
        )
        httpd_thread.setDaemon(True)
        httpd_thread.start()
        yield s


# @pytest.fixture(scope='function')
# def sample_server():
#     with requests_mock.mock() as m:
#         m.post(
#             "http://127.0.0.1:{}".format(SERVER_PORT),
#             text=json.dumps(DEFAULT_RESPONSE)
#         )
#         yield m


@pytest.fixture(scope='function')
def sample_client():
    return client.KelnerClient('http://127.0.0.1:%d' % (SERVER_PORT))
