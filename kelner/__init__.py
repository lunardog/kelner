from . import server
from . import client
from . import model


def serve(kelner_model, host="127.0.0.1", port=server.KELNER_PORT):
    """
    Serves the loaded kelner_model
    """
    k_server = server.KelnerServer(kelner_model)
    return k_server.serve_http(host, port)
