# coding=utf-8
# Copyright 2022, LIU Jialin.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# pylint: disable=ungrouped-imports

"""Bidirectional Conditional GANs."""

import os
import math

from datetime import datetime
from absl import logging

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
from tensorflow.keras.losses import BinaryCrossentropy


try:
    # Module import
    from .context import _DATADIR, _WORKDIR, main
except ImportError:
    # Script import
    from context import _DATADIR, _WORKDIR, main


class BiCoGAN(Model):
    """Invariant BiCoGAN (https://arxiv.org/pdf/1711.07461v1.pdf)
    """
    def __init__(self, name='bicogan_v1', **kwargs):
        super().__init__(name=name, **kwargs)
        # Image size
        self.h, self.w, self.c = 28, 28, 1
        self.img_shape = self.h * self.w * self.c
        self.n_dim = 10

        # Training details
        self.lr = 1e-4
        self.epochs = 400
        self.batch_size = 128

        # Latent dim
        self.z_dim = 50

        # EFL weight
        self.gamma = 5

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

        # Working directory
        timeshift = datetime.today().strftime('%Y%m%d-%H:%M:%S')

        self.workdir = os.path.join(_WORKDIR, f'{name}/{timeshift}')
        self.logdir = os.path.join(self.workdir, 'logs')
        self.imgdir = os.path.join(self.workdir, 'imgs/')

        tf.io.gfile.makedirs(self.workdir)
        tf.io.gfile.makedirs(self.logdir)
        tf.io.gfile.makedirs(self.imgdir)

        # Save model architecture, not weights.
        # MNIST is a small dataset to reproduce.
        gjson = self.generator.to_json()
        ejson = self.encoder.to_json()
        djson = self.discriminator.to_json()

        with open(
            os.path.join(self.workdir, 'model.json'),
            mode='w',
            encoding='utf-8',
        ) as json_file:
            json_file.write(gjson)
            json_file.write(ejson)
            json_file.write(djson)

        # Keras Callbacks.
        self.callbacks = [
            SaveImage(workdir=self.imgdir, latent_dim=self.z_dim),
            LearningRateDecay(init_lr=self.lr, decay_steps=self.epochs / 2.0),
            keras.callbacks.TensorBoard(log_dir=self.logdir),
        ]

        # Optimizer
        self.d_optimizer = Adam(
            learning_rate=self.lr, beta_1=0.5, beta_2=0.999
        )
        self.g_optimizer = Adam(
            learning_rate=self.lr, beta_1=0.5, beta_2=0.999
        )
        self.e_optimizer = Adam(
            learning_rate=self.lr, beta_1=0.5, beta_2=0.999
        )

        # External factor loss: BCE
        self.efl = BinaryCrossentropy(from_logits=True, name='EFL')

        # Metric monitor
        self.d_loss_metric = keras.metrics.Mean(name='d_loss')
        self.e_loss_metric = keras.metrics.Mean(name='e_loss')
        self.lr_metric = Monitor(name='lr')

    @property
    def metrics(self):
        return [
            self.d_loss_metric,
            self.e_loss_metric,
            self.lr_metric
        ]

    def train_step(self, data):
        """One step forward / backward computation.
        """
        # Unpack data
        img, c = data
        batch_size = tf.shape(img)[0]

        img = tf.reshape(
            img,
            shape=(batch_size, -1),
            name='flatimg',
        )
        z = tf.random.uniform(
            shape=(batch_size, self.z_dim),
            minval=-1,
            maxval=1,
            dtype=tf.float32
        )

        with tf.GradientTape(persistent=True) as tape:
            # Forward pass: Encoder and Generator
            z_, c_ = self.encoder(img)
            img_ = self.generator([z, c])
            # Forward pass: Discriminator
            d_inputs = [
                tf.concat([img_, img], 0),
                tf.concat([z, z_], 0),
                tf.concat([c, c_], 0),
            ]
            d_outs = self.discriminator(d_inputs)
            pred_g, pred_e = tf.split(d_outs, num_or_size_splits=2, axis=0)

            # Loss function
            d_loss = tf.reduce_mean(tf.math.softplus(pred_g)) + \
                tf.reduce_mean(tf.math.softplus(-pred_e))
            g_loss = tf.reduce_mean(tf.math.softplus(-pred_g))
            e_loss = tf.reduce_mean(tf.math.softplus(pred_e)) + \
                self.gamma * self.efl(c_, c)

        # Gradient computation
        d_gradients = tape.gradient(
            d_loss, self.discriminator.trainable_variables,
        )
        g_gradients = tape.gradient(
            g_loss, self.generator.trainable_variables,
        )
        e_gradients = tape.gradient(
            e_loss, self.encoder.trainable_variables,
        )

        # Release tape object
        del tape

        # Optimizer update
        self.d_optimizer.apply_gradients(
            zip(d_gradients, self.discriminator.trainable_variables)
        )
        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )
        self.e_optimizer.apply_gradients(
            zip(e_gradients, self.encoder.trainable_variables)
        )

        # Metric update
        self.d_loss_metric.update_state(d_loss)
        self.e_loss_metric.update_state(g_loss)
        self.lr_metric.update_state(self.d_optimizer.learning_rate)

        return {
            'd_loss': self.d_loss_metric.result(),
            'e_loss': self.e_loss_metric.result(),
            'lr': self.lr_metric.result(),
        }

class SaveImage(keras.callbacks.Callback):
    """Save image: callback function.
    """
    def __init__(self, workdir='.', interval=10, latent_dim=50):
        self.num_img = 100
        self.latent_dim = latent_dim
        self.workdir = workdir
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None): # pylint: disable=unused-argument
        """Save at the end of #interval epochs.
        """
        if not tf.math.mod(epoch + 1, self.interval):
            # Latent vector
            z = tf.random.uniform(
                shape=(self.num_img, self.latent_dim),
                minval=-1.,
                maxval=1.
            )
            # Label vector
            labels = tf.range(0, 10, dtype=tf.int32)
            labels = tf.reshape(labels, shape=(1, -1))
            labels = tf.tile(labels, [1, 10])
            c = tf.one_hot(labels, depth=10)[0]

            # Image generation
            image = self.model.generator.predict([z, c])

            # Save in png format
            # image = np.reshape(image, (10, 10, 28, 28))
            # image = np.transpose(image, (0, 2, 1, 3))
            # image = np.reshape(image, (10 * 28, 10 * 28))
            nrow = 10
            padding = 2
            pad_value = 0.3

            ndarray = np.reshape(image, (100, 28, 28, 1))
            ndarray = np.concatenate((ndarray, ndarray, ndarray), -1)

            nmaps = ndarray.shape[0]
            xmaps = min(nrow, nmaps)
            ymaps = int(math.ceil(float(nmaps) / xmaps))
            height = int(ndarray.shape[1] + padding)
            width = int(ndarray.shape[2] + padding)
            num_channels = ndarray.shape[3]

            grid = np.full(
                (
                    height * ymaps + padding,
                    width * xmaps + padding,
                    num_channels
                ),
                pad_value,
            ).astype(np.float32)

            k = 0
            for y in range(ymaps):
                for x in range(xmaps):
                    if k >= nmaps:
                        break
                    grid[
                        y * height + padding:(y + 1) * height,
                        x * width + padding:(x + 1) * width
                    ] = ndarray[k]
                    k += 1

            image = 255 * (grid + 1) / 2
            image = np.clip(image, 0, 255)
            image = image.astype('uint8')

            tf.io.gfile.makedirs(self.workdir)
            Image.fromarray(image.copy()).save(
                os.path.join(self.workdir, f'G_z-{epoch + 1}.png')
            )

class LearningRateDecay(keras.callbacks.Callback):
    """Rate decay schedule: exponential decay.
    """
    def __init__(self, init_lr, decay_steps):
        self.lr = init_lr
        self.decay_rate = 0.1
        self.bound = decay_steps

    def on_epoch_begin(self, epoch, logs=None): # pylint: disable=unused-argument
        d_lr = self.model.d_optimizer.learning_rate
        g_lr = self.model.g_optimizer.learning_rate
        e_lr = self.model.e_optimizer.learning_rate
        if epoch >= self.bound:
            weight = self.decay_rate ** (epoch / self.bound)
            d_lr.assign(self.lr * weight)
            g_lr.assign(self.lr * weight)
            e_lr.assign(self.lr * weight)

class Monitor(keras.metrics.Metric):
    """Learning rate monitor.
    """
    def __init__(self, name='lr_monitor', **kwargs):
        super().__init__(name=name, **kwargs)
        self.lr = tf.Variable(0.0, trainable=False)

    def update_state(self, lr):
        self.lr.assign(lr)

    def result(self):
        return self.lr

    def reset_state(self):
        self.lr.assign(0.0)

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
    x = Input(shape=(img_shape,), name='image')
    z = Input(shape=(z_dim,), name='intrinsic_z')
    c = Input(shape=(n_dim,), name='extrinsic_c')
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

    ds_train = ds_train.map(
        norm_and_remove,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(
        norm_and_remove,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_test = ds_test.cache()
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return (ds_train, ds_test)

def test1(argv):
    """DEBUG: Network Summary."""
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

@main
def run(argv):
    """Train BiCoGAN on MNIST."""
    del argv

    # Environment variable setting
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['NO_GCE_CHECK'] = 'true'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    logging.set_verbosity(logging.DEBUG)

    bicogan = BiCoGAN()

    ds_train, _ = get_mnist(batch_size=bicogan.batch_size)

    bicogan.compile(
        run_eagerly=False,
    )

    bicogan.fit(
        ds_train,
        epochs=bicogan.epochs,
        callbacks=bicogan.callbacks,
    )
