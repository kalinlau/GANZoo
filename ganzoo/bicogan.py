""" Bidirectional Conditional GAN (https://arxiv.org/abs/1711.07461) """
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['NO_GCE_CHECK'] = 'true'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from absl import logging
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
from tensorflow.keras.layers import ReLU, BatchNormalization, Reshape
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist


'''Import at different mode

Module mode:
    >>> import ganzoo
    >>> from ganzoo import bicogan
Script mode:
    >>> python bicogan.py
'''
try:
    # Module import
    # Raise `ModuleImportError` without '.' relative dot.
    from .context import _DATADIR, _WORKDIR, main
except:
    # Script import
    from context import _DATADIR, _WORKDIR, main


__all__ = ['BiCoGAN']

class BiCoGAN(Model):
    """Invariant BiCoGAN."""
    def __init__(self, name='bicogan', **kwargs):
        super().__init__(name=name, **kwargs)
        # Image size
        self.H, self.W, self.C = 28, 28, 1
        self.img_shape = self.H * self.W * self.C
        self.n_dim = 10

        # Training details
        self.lr = 1e-4
        self.epochs = 400
        self.batch_size = 128

        # Latent dim
        self.z_dim = 50

        # Build models
        self.encoder = build_encoder(
            self.img_shape, self.n_dim, self.z_dim
        )
        self.generator = build_generator(
            self.img_shape, self.n_dim, self.z_dim
        )
        self.discriminator = build_discriminator(
            self.img_shape, self.n_dim, self.z_dim
        )

        # Callbacks
        timeshift = datetime.today().strftime('%Y%m%d-%H:%M:%S')

        self.workdir = os.path.join(_WORKDIR, f'{name}/{timeshift}')
        self.logdir = os.path.join(self.workdir, 'logs')
        self.imgdir = os.path.join(self.workdir, 'imgs/')

        # tf.io.gfile.makedirs(self.workdir)
        # tf.io.gfile.makedirs(self.logdir)
        # tf.io.gfile.makedirs(self.imgdir)


def build_encoder(
        img_shape,
        n_dim,
        z_dim,
        reg = lambda: L1L2(l1=0., l2=2.5e-5),
    ):
    """Encoder: z, c = E(x; theta_E)

    Inverse mapping x (image) back to both intrinsic factor z (latent vector)
    and extrinsic factor c (label)

    WARNING: API is not same as mentioned in paper,  output two factors instead
    of a concatenated one.
    """
    x = Input(shape=(img_shape), name='image')
    y = Dense(
        1024,
        kernel_regularizer=reg(),
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)
    )(x)
    y = LeakyReLU(alpha=0.2)(y)
    y = Dense(
        1024,
        kernel_regularizer=reg(),
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)
    )(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = BatchNormalization()(y)

    z = Dense(
        z_dim,
        kernel_regularizer=reg(),
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
        name='out_z',
    )(y)
    c = Dense(
        n_dim,
        kernel_regularizer=reg(),
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
        name='out_c',
    )(y)

    return Model(x, [z, c], name='encoder')

def build_generator(
    img_shape,
    n_dim,
    z_dim,
    reg = lambda: L1L2(l1=0., l2=2.5e-5)
):
    """Generator: x = G([z c]; theta_G)

    Generate Image x by intrinsic factor z (latent vector) and
    extrinsic factor c (label)
    """
    z = Input(shape=(z_dim), name='intrinsic_z')
    c = Input(shape=(n_dim), name='extrinsic_c')
    h = Concatenate(axis=-1)([z, c])
    y = Dense(
        1024,
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
        kernel_regularizer=reg(),
    )(h)
    y = ReLU()(y)
    y = Dense(
        1024,
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
        kernel_regularizer=reg(),
    )(y)
    y = ReLU()(y)
    y = BatchNormalization()(y)
    x = Dense(
        img_shape,
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
        kernel_regularizer=reg(),
        name='out',
    )(y)
    return Model([z, c], x, name='generator')


def build_discriminator(
    img_shape,
    n_dim,
    z_dim,
    reg = lambda: L1L2(l1=0., l2=2.5e-5),
):
    """Discriminator: D(x, z, c; theta_D)

    Predict whether image x and latent vector is real or fake.
    No `sigmoid` activation at output layer

    fake: D(G(z_hat), z_hat; theta_D), z_hat = [z c]
    Real: D(x, E(x); theta_D)
    """
    x = Input(shape=(img_shape), name='image')
    z = Input(shape=(z_dim), name='intrinsic_z')
    c = Input(shape=(n_dim), name='extrinsic_c')
    y = Concatenate(axis=-1)([x, z, c])

    y = Dense(
        1024,
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.5),
        kernel_regularizer=reg(),
    )(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = Dense(
        1024,
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
        kernel_regularizer=reg(),
    )(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = BatchNormalization()(y)
    y = Dense(
        1,
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
        kernel_regularizer=reg(),
        name='out',
    )(y)
    return Model([x, z, c], [y], name='discriminator')


def get_mnist(data_dir=_DATADIR, batch_size=128):
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
@main
def main(argv):
    del argv
    h, w, c = 28, 28, 1
    img_shape = h * w * c
    n_dim = 10
    z_dim = 50

    encoder = build_encoder(img_shape, n_dim, z_dim)
    encoder.summary()

    generator = build_generator(img_shape, n_dim, z_dim)
    generator.summary()

    discriminator = build_discriminator(img_shape, n_dim, z_dim)
    discriminator.summary()


    logging.debug(discriminator.get_layer('out').output.shape)