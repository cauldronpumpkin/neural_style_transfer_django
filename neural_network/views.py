from .serializers import NeuralNetSerializer
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import FileUploadParser
import cloudinary
import cloudinary.uploader
import base64
import tensorflow_hub as hub
import functools
import time
import os
import PIL.Image
import numpy as np
from django.shortcuts import render
import tensorflow as tf
import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False


def get_tensors_from_images(data):

    max_dim = 1024

    img0 = tf.io.read_file('./' + data['main_image'])
    img0 = tf.image.decode_image(img0, channels=3)
    img0 = tf.image.convert_image_dtype(img0, tf.float32)

    shape = tf.cast(tf.shape(img0)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img0 = tf.image.resize(img0, new_shape)
    img0 = img0[tf.newaxis, :]

    img1 = tf.io.read_file('./' + data['style_image'])
    img1 = tf.image.decode_image(img1, channels=3)
    img1 = tf.image.convert_image_dtype(img1, tf.float32)

    shape = tf.cast(tf.shape(img1)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img1 = tf.image.resize(img1, new_shape)
    img1 = img1[tf.newaxis, :]

    return (img0, img1)


def tensor_to_image(tensor):

    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)

    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]

    return PIL.Image.fromarray(tensor)


class FileUploadView(APIView):
    # parser_class = (FileUploadParser,)

    def post(self, request, *args, **kwargs):
        file_serializer = NeuralNetSerializer(data=request.data)

        if file_serializer.is_valid():
            file_serializer.save()

            (main_image, style_image) = get_tensors_from_images(
                file_serializer.data)
            hub_module = hub.load(
                'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
            stylized_image = hub_module(tf.constant(
                main_image), tf.constant(style_image))[0]

            res = tensor_to_image(stylized_image)
            path = './media/{}.jpg'.format(time.time())
            res.save(path)
            context = cloudinary.uploader.upload(path,
                                                 folder="/neural_style_transfer/",
                                                 )['url']
            for file in os.listdir('./media'):
                os.remove('./media/' + file)

            return Response(context, status=status.HTTP_201_CREATED)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# Create your views here.
