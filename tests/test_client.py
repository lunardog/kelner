from __future__ import absolute_import

import os
import json

from .fixtures import sample_client, sample_server, sample_file
from . import fixtures

from kelner import client


def test_decode_file(sample_client, sample_file):
    raw_data, mimetype, size = client.decode_file(sample_file)
    assert mimetype == 'application/json'
    assert size > 0
    json_data = json.loads(raw_data)
    assert json_data == fixtures.SAMPLE_DATA
    if os.path.isfile(sample_file):
        os.unlink(sample_file)


def test_classify(sample_server, sample_client, sample_file):
    scores = sample_client.classify(sample_file, fixtures.DEFAULT_LABELS)
    for label, score in scores:
        assert score == 1.0
        assert label == 'b'


def test_request(sample_server, sample_client):
    sample_data = json.dumps(fixtures.SAMPLE_DATA)
    l = sample_client.request(sample_data)
    assert l.decode('utf-8') == str(fixtures.DEFAULT_RESPONSE)
