import io
from sys import version_info
import email
import boto3
from keras.utils.data_utils import get_file as keras_get_file

try:  # python3
    from urllib.request import BaseHandler, URLError, url2pathname, addinfourl
    from urllib.request import build_opener, install_opener

except:  # python2
    from urllib2 import BaseHandler, URLError, url2pathname, addinfourl
    from urllib2 import build_opener, install_opener


def get_file(uri, extract=False):

    if '://' not in uri:
        return uri
        # uri = 'file://' + uri

    fname = uri.split('/')[-1]
    local_path = keras_get_file(
        fname, uri,
        extract=extract,
        cache_subdir='models')

    return local_path


class _FileLikeKey(io.BufferedIOBase):

    def __init__(self, key):
        self.read = key.read


class S3Handler(BaseHandler):

    def s3_open(self, req):
        # The implementation was inspired mainly by the code behind
        # urllib.request.FileHandler.file_open().
        #
        # recipe copied from:
        # http://code.activestate.com/recipes/578957-urllib-handler-for-amazon-s3-buckets/
        # converted to boto3

        if version_info[0] < 3:
            bucket_name = req.get_host()
            key_name = url2pathname(req.get_selector())[1:]
        else:
            bucket_name = req.host
            key_name = url2pathname(req.selector)[1:]

        if not bucket_name or not key_name:
            raise URLError('url must be in the format s3://<bucket>/<key>')

        s3 = boto3.resource('s3')

        key = s3.Object(bucket_name, key_name)

        client = boto3.client('s3')
        obj = client.get_object(Bucket=bucket_name, Key=key_name)
        filelike = _FileLikeKey(obj['Body'])

        origurl = 's3://{}/{}'.format(bucket_name, key_name)

        if key is None:
            raise URLError('no such resource: {}'.format(origurl))

        headers = [
            ('Content-type', key.content_type),
            ('Content-encoding', key.content_encoding),
            ('Content-language', key.content_language),
            ('Content-length', key.content_length),
            ('Etag', key.e_tag),
            ('Last-modified', key.last_modified),
        ]

        headers = email.message_from_string(
            '\n'.join('{}: {}'.format(key, value) for key, value in headers
                      if value is not None))
        return addinfourl(filelike, headers, origurl)


opener = build_opener(S3Handler)
install_opener(opener)
