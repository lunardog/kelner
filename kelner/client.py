import requests
import json
import yaml
import magic
import os


def dict_to_yaml(dictionary):
    yaml_lines = ['%s: %f' % (key, dictionary[key]) for key in dictionary]
    return os.linesep.join(yaml_lines)


def get_label(labels, index):
    if index < len(labels):
        return labels[index]
    else:
        return '#%d' % (index)


def attach_labels(vector, labels):
    """
    Creates a list of (label, score) tuples
    from a vector of scores and list of labels
    """
    labelled = [
        (get_label(labels, index), value)
        for index, value in enumerate(vector)
    ]
    labelled.sort(key=lambda x: x[1], reverse=True)
    return labelled


def decode_file(path):
    size = os.path.getsize(path)
    mimetype = magic.from_file(path, mime=True)
    with open(path, 'rb') as file:
        contents = file.read(size)

    # if detected as text/plain, try opening as json
    if mimetype == 'text/plain':
        try:
            contents = contents.decode('utf-8')
            json.loads(contents)
            mimetype = 'application/json'
        except:
            pass
    return contents, mimetype, size


class KelnerClient(object):

    def __init__(self, url=None):
        if url is None:
            url = 'http://127.0.0.1:%d' % (0xf00d)
        self.url = url
        self.output_format = 'raw'
        self.labels = []

    def format_output(
            self,
            response,
            output_format='json',
            labels=None,
            sort_values=False
    ):
        if labels is not None:
            labelled = attach_labels(response[0], labels, sort_values)

            if output_format == 'yaml':
                response = dict_to_yaml(labelled)
            else:
                response = json.dumps(
                    labelled,
                    indent=2,
                    sort_keys=False
                )

        else:
            if output_format == 'yaml':
                response = yaml.dump(response)
            else:
                response = json.dumps(response, indent=2, sort_keys=False)

        return response

    def classify(self, file_path, labels=[], top=1):
        data, mimetype, size = decode_file(file_path)
        response = self.request(data, mimetype, size)
        data = json.loads(response.decode('utf-8'))
        scores = attach_labels(data[0], labels)
        return scores[0:top]

    def request(self, data, mimetype='application/json', size=None):
        try:
            encoded = bytes(str(data), 'utf-8')
        except:
            encoded = bytes(str(data))
        if size is None:
            size = len(encoded)
        headers = {'Content-type': mimetype, 'Content-Length': str(size)}
        res = requests.post(self.url, data=data, headers=headers)
        return res.content
