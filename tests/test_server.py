from __future__ import absolute_import

import requests

from .fixtures import sample_echo_server
from . import fixtures


def test_get(sample_echo_server):
    r = requests.get(
        sample_echo_server.url
    )
    assert r.status_code == 200


def test_empty_post(sample_echo_server):
    headers = {'Content-Length': 0}
    try:
        r = requests.post(
            sample_echo_server.url,
            headers=headers
        )
        assert r.status == 411
    except:
        pass


def test_post(sample_echo_server):
    data = '[1.0, 2.0, 3.0]'
    headers = {
        'Content-Type': 'application/json',
        'Content-Length': str(len(data))
    }
    r = requests.post(
        sample_echo_server.url,
        data=data, headers=headers
    )
    assert r.status_code == 200
    # The dummy model should echo the original data back
    assert str(r.json()) == data
