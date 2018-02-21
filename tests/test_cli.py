from kelner import cli

import os
from .fixtures import runner
from .fixtures import sample_model


def test_kelnerd(runner, sample_model):
    model_file_name = '/tmp/my_model.h5'
    with runner.isolated_filesystem():
        sample_model.save(model_file_name)
        try:
            result = runner.invoke(
                cli.kelnerd,
                ['--load-model', model_file_name, '--dry-run']
            )
            assert not result.exception
            assert 'Listening' in result.output
        except Exception as e:
            assert False, 'exception during server test '+str(e)
            print(e)
        finally:
            os.remove(model_file_name)


def test_kelner(runner):
    result = runner.invoke(cli.kelner)
    assert result.exit_code == 0
    assert not result.exception
