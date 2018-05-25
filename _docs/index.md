---
# You don't need to edit this file, it's empty on purpose.
# Edit theme's home layout instead if you wanna make some changes
# See: https://jekyllrb.com/docs/themes/#overriding-theme-defaults
---

## What is Kelner?

Ridiculously simple model serving.

1. Get an exported model (download or train and save)
2. `kelnerd -m SAVED_MODEL_FILE`
3. There is no step 3, your model is served

## Quickstart

### Install `kelner`

```bash
    $ pip install kelner
```

### Download a Tensorflow ProtoBuff file

```bash
    $ wget https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip
    $ unzip inception_dec_2015.zip
        Archive:  inception_dec_2015.zip
        inflating: imagenet_comp_graph_label_strings.txt
        inflating: LICENSE
        inflating: tensorflow_inception_graph.pb
    $ kelnerd -m tensorflow_inception_graph.pb --engine tensorflow --input-node ExpandDims --output-node softmax
```

### Run the server

```bash
    $ kelnerd -m tensorflow_inception_graph.pb --engine tensorflow --input-node ExpandDims --output-node softmax
```

### Send a request to the model:

```bash
    $ curl --data-binary "@dog.jpg" localhost:61453 -X POST -H "Content-Type: image/jpeg"
```

The response should be a JSON-encoded array of floating point numbers.

For a fancy client (not really necessary, but useful) you can use the `kelner` command.

This is how you get the top 5 labels from the server you ran above (note the `head -n 5` part):

```bash
    $ kelner classify dog.jpg --imagenet-labels --top 5
    boxer: 0.973630
    Saint Bernard: 0.001821
    bull mastiff: 0.000624
    Boston bull: 0.000486
    Greater Swiss Mountain dog: 0.000377
```

## FAQ

### Who is this for?

Machine learning researchers who don't want to deal with building a web server for every model they export.

Kelner loads a saved Keras or Tensorflow model and starts an HTTP server that pipes POST request body to the model and returns JSON-encoded model response.

###  How is it different from Tensorflow Serving?

1. Kelner is ridiculously simple to install and run
2. Kelner also works with saved Keras models
3. Kelner works with one model per installation
4. Kelner doesn't do model versioning
5. Kelner is JSON over HTTP while tf-serving is ProtoBuf over gRPC
5. Kelner's protocol is:
    * `GET` returns model input and output specs as JSON
    * `POST` expects JSON or an image file, returns JSON-encoded result of model inference
