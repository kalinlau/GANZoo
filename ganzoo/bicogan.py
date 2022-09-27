""" Bidirectional Conditional GAN (https://arxiv.org/abs/1711.07461) """
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['NO_GCE_CHECK'] = 'true'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from absl import app, logging
from absl.logging import debug, info
logging.set_verbosity(logging.DEBUG)

import sys
from datetime import datetime

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate, Dense, LeakyReLU
from tensorflow.keras.layers import ReLU, BatchNormalization
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist


class BiCoGAN(Model):
    def __init__(self, name='bicogan', **kwargs):
        super().__init__(name=name, **kwargs)
        # Image size

        # Latent dim

        # Build models

        # Callbacks


def build_encoder():
    """Encoder Network."""
    pass

def build_generator():
    pass

def build_discriminator():
    pass


def get_mnist(data_dir='./data', batch_size=128):
    def norm_and_remove(img, label):
        """Normalize to [-1, 1] and Remove label

        Args:
            img (tf.Tensor): mnist image tensor
            label (tf.float32): mnist image label

        Returns:
            img: normalized image
        """
        return (
            (tf.cast(img, tf.float32) - 127.5) / 127.5,
            tf.one_hot(label, depth=10),
        )

    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        with_info=True,
        as_supervised=True,
        data_dir=data_dir,
        try_gcs=False,
    )

    ds_train = ds_train.map(norm_and_remove, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(norm_and_remove, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.cache()
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return (ds_train, ds_test)


# Main routine
# @app.run
def main(argv):
    del argv

class TestDatasetMNIST:
    def test_data_set(argv):
        del argv

        # Debug speedup: use cpu
        tf.config.set_visible_devices([], 'GPU')

        ds_train, ds_test = get_mnist(batch_size=5)

        for img, label in ds_train.take(1):
            assert img.shape == (5, 28, 28, 1)
            assert label.shape == (5, 10)
            assert -1 <= img.numpy().any() <= 1

            debug(img.shape)
            debug(label.shape)