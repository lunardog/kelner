import io
import numpy as np
from PIL import Image
import json
import requests

try:
    # python 3
    import http.server as http
except:
    # python 2
    import BaseHTTPServer as http

KELNER_PORT = 0xf00d


class HTTPHandler(http.BaseHTTPRequestHandler):

    kelner_server = None

    def reply(self, response):
        """ Sends the JSON-encoded response to the client """
        message = response
        self.send_response(requests.codes.ok)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        try:
            self.wfile.write(bytes(message, 'utf8'))
        except:
            self.wfile.write(bytes(message))

    def extract_content(self):
        # check content type
        content_length = self.headers['Content-Length']
        if content_length is None or content_length == 0:
            return self.send_error(requests.codes.length_required)

        content_length = int(content_length)

        content = self.rfile.read(content_length)
        mimetype = self.headers['Content-Type']

        if mimetype is None:
            return self.send_error(
                requests.codes.bad,
                message=str('Unknown Content Type')
            )
        return content, mimetype

    def do_GET(self):
        """ Returns model parameters """

        model_info = self.kelner_server.get_model_info()
        self.reply(model_info)
        return

    def do_POST(self):
        """ Takes care of incoming data to infer """

        content, mimetype = self.extract_content()

        try:
            message = self.kelner_server.process_content(content, mimetype)
            return self.reply(message)

        except ValueError as err:
            return self.send_error(requests.codes.bad, message=str(err))


class KelnerServer():

    def __init__(self, model):
        self.model = model
        self.server = None
        self.port = None
        self.url = None

    def __del__(self):
        if self.server is not None:
            self.server.server_close()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        if self.server is not None:
            self.server.server_close()

    def serve_http(self, host, port):
        HTTPHandler.kelner_server = self
        http.HTTPServer.allow_reuse_address = 1
        self.host = host
        self.port = port
        self.url = 'http://{}:{}'.format(self.host, self.port)
        self.server = http.HTTPServer(
            (host, port),
            HTTPHandler
        )
        self.server.serve_forever()

    def tensor_to_json(self, tensor, pretty=False):
        arr = tensor.astype(float).tolist()
        if pretty:
            return json.dumps(arr, indent=2, sort_keys=False)
        else:
            return json.dumps(arr)

    def get_model_info(self):
        """ Returns information about inputs and outputs of the model """
        dimension_values = lambda dim: tuple(d.value for d in dim)
        info = {}
        if self.model.input is not None:
            info['in'] = {
                'name': self.model.input.name,
                'shape': dimension_values(self.model.input.shape)
            }
        if self.model.output is not None:
            info['out'] = {
                'name': self.model.output.name,
                'shape': dimension_values(self.model.output.shape)
            }
        if self.model.labels is not None:
            info['labels'] = self.model.labels
        return info

    def process_content(self, content, mimetype):
        """ Processes given content """
        data = self.extract_data(content, mimetype)
        inference = self.model(data)
        message = self.tensor_to_json(inference)
        return message

    def extract_data(self, content, mimetype):
        """ Turns content into a numpy array, taking mimetype into account """
        if mimetype.startswith('image/'):
            # Load image
            stream = io.BytesIO(content)
            img = Image.open(stream)
            data = np.array(img)[np.newaxis]
        elif mimetype == 'application/json':
            # Load json
            data = json.loads(content.decode())
            data = np.array(data, np.float32)
        else:
            # Otherwise, just load bytes
            data = bytes(content)
        return data
