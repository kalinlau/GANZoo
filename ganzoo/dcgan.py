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

"""Vanilla GANs."""

import os
import math

from datetime import datetime
from absl import logging, flags

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds


try:
    # Module import
    from .context import _DATADIR, _WORKDIR, main
except ImportError:
    # Script import
    from context import _DATADIR, _WORKDIR, main

FLAGS = flags.FLAGS
FLAGS.alsologtostderr = True

flags.DEFINE_boolean(
    'run_eagerly', False, 'Run in Eager mode to debug. (Default: False)',
)


class DCGAN(keras.Model):
    def __init__(self, name='dcgan', **kwargs):
        super().__init__(name=name, **kwargs)
        # Image
        self.h, self.w, self.c = 28, 28, 1
        self.img_shape = (self.h, self.w, self.c)

        # Training details
        self.batch_size = 64
        self.epochs = 40
        self.lr = 1e-4
        self.z_dim = 100
        self.beta_1 = 0.5

        # Model
        self.generator = build_generator(self.z_dim)
        self.discriminator = build_discriminator(self.img_shape)

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
        djson = self.discriminator.to_json()

        with open(
            os.path.join(self.workdir, 'model.json'),
            mode='w',
            encoding='utf-8',
        ) as json_file:
            json_file.write(gjson)
            json_file.write(djson)


        # Keras Callbacks.
        self.callbacks = [
            SaveImage(workdir=self.imgdir, latent_dim=self.z_dim, interval=2),
            LearningRateDecay(init_lr=self.lr, decay_steps=self.epochs / 2.0),
            keras.callbacks.TensorBoard(log_dir=self.logdir),
        ]

        # Optimziers
        self.d_optimizer = keras.optimizers.Adam(
            learning_rate= self.lr, beta_1=self.beta_1
        )
        self.g_optimizer = keras.optimizers.Adam(
            learning_rate= self.lr, beta_1=self.beta_1
        )

        self.d_loss_metric = keras.metrics.Mean(name='d_loss')
        self.g_loss_metric = keras.metrics.Mean(name='g_loss')
        self.lr_metric = Monitor(name='d_lr')

        # Loss
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)

    @property
    def metrics(self):
        return [
            self.d_loss_metric,
            self.lr_metric,
        ]

    def train_step(self, data):
        img = data
        batch_size = tf.shape(img)[0]

        # Train Discriminator
        z = tf.random.normal(shape=(batch_size, self.z_dim))
        real = tf.zeros((batch_size, 1))
        fake = tf.ones((batch_size, 1))
        labels = tf.concat([fake, real], 0)
        # Important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))
        with tf.GradientTape() as tape:
            img_ = self.generator(z)
            imgs = tf.concat([img_, img], 0)

            d_outs = self.discriminator(imgs)
            dloss = self.loss_fn(labels, d_outs)

        d_grad = tape.gradient(
            dloss, self.discriminator.trainable_weights
        )
        self.d_optimizer.apply_gradients(
            zip(d_grad, self.discriminator.trainable_weights)
        )

        # Train Generator
        z = tf.random.normal(shape=(batch_size, self.z_dim))
        labels = tf.zeros((batch_size, 1))
        with tf.GradientTape() as tape:
            img_ = self.generator(z)
            d_out = self.discriminator(img_)

            gloss = self.loss_fn(labels, d_out)

        g_grad = tape.gradient(
            gloss, self.generator.trainable_weights
        )
        self.g_optimizer.apply_gradients(
            zip(g_grad, self.generator.trainable_weights)
        )

        self.d_loss_metric.update_state(dloss)
        self.g_loss_metric.update_state(gloss)
        self.lr_metric.update_state(self.d_optimizer.learning_rate)


        return {
            'dloss': self.d_loss_metric.result(),
            'lr': self.lr_metric.result(),
        }


def build_generator(z_dim):
    # z = keras.layers.Input(shape=z_dim)
    # y = keras.layers.Dense(7 * 7 * z_dim)(z)
    # y = keras.layers.LeakyReLU(alpha=0.2)(y)
    # y = keras.layers.Reshape((7, 7, z_dim))(y)
    # y = keras.layers.Conv2DTranspose(
    #     128, kernel_size=4, strides=2, padding='same'
    # )(y)
    # y = keras.layers.LeakyReLU(alpha=0.2)(y)
    # y = keras.layers.Conv2DTranspose(
    #     128, kernel_size=4, strides=2, padding='same'
    # )(y)
    # y = keras.layers.LeakyReLU(alpha=0.2)(y)
    # y = keras.layers.Conv2D(
    #     1, kernel_size=7, padding='same', activation='sigmoid',
    # )(y)

    # Same architecture with infoGAN
    # z = keras.layers.Input(shape=(z_dim,))
    # y = keras.layers.Dense(1024)(z)
    # y = keras.layers.BatchNormalization(epsilon=1e-5)(y)
    # y = keras.layers.ReLU()(y)
    # y = keras.layers.Dense(7 * 7 * 128)(y)
    # y = keras.layers.BatchNormalization(epsilon=1e-5)(y)
    # y = keras.layers.ReLU()(y)
    # y = keras.layers.Reshape((7, 7, 128))(y)
    # y = keras.layers.Conv2DTranspose(64, (4,4), (2,2), padding='same')(y)
    # y = keras.layers.BatchNormalization(epsilon=1e-5)(y)
    # y = keras.layers.ReLU()(y)
    # y = keras.layers.Conv2DTranspose(
    #     1, (4,4), (2,2), padding='same', activation='sigmoid'
    # )(y)
    z = keras.layers.Input(shape=z_dim)
    y = keras.layers.Dense(
        128 * 7 * 7, activation='relu'
    )(z)
    y = keras.layers.Reshape((7, 7, 128))(y)
    y = keras.layers.UpSampling2D()(y)
    y = keras.layers.Conv2D(
        128, kernel_size=3, padding='same'
    )(y)
    y = keras.layers.BatchNormalization(momentum=0.8)(y)
    y = keras.layers.ReLU()(y)
    y = keras.layers.UpSampling2D()(y)
    y = keras.layers.Conv2D(
        64, kernel_size=3, padding='same'
    )(y)
    y = keras.layers.BatchNormalization(momentum=0.8)(y)
    y = keras.layers.ReLU()(y)
    y = keras.layers.Conv2D(
        1, kernel_size=3, padding='same', activation='tanh'
    )(y)

    return keras.Model(z, y, name='generator')

def build_discriminator(img_shape):
    # img = keras.layers.Input(shape=img_shape) # (H, W, C)
    # y = keras.layers.Conv2D(
    #     64, kernel_size=4, strides=2, padding="same")(img)
    # y = keras.layers.LeakyReLU(alpha=0.2)(y)
    # y = keras.layers.Conv2D(
    #     128, kernel_size=4, strides=2, padding="same")(y)
    # y = keras.layers.LeakyReLU(alpha=0.2)(y)
    # y = keras.layers.GlobalMaxPooling2D()(y)
    # y = keras.layers.Dense(1, activation="sigmoid")(y)

    # Same architecture with infoGAN
    # img = keras.layers.Input(shape=img_shape)
    # y = keras.layers.Conv2D(64, (4,4), (2,2), padding='same')(img)
    # y = keras.layers.LeakyReLU(alpha=0.2)(y)
    # y = keras.layers.Conv2D(128, (4,4), (2,2), padding='same')(y)
    # y = keras.layers.BatchNormalization(epsilon=1e-5)(y)
    # y = keras.layers.LeakyReLU(alpha=0.2)(y)
    # y = keras.layers.Flatten()(y)
    # y = keras.layers.Dense(1024)(y)
    # y = keras.layers.BatchNormalization(epsilon=1e-5)(y)
    # y = keras.layers.LeakyReLU(alpha=0.2)(y)
    # y = keras.layers.Dense(1, activation='sigmoid')(y)
    img = keras.layers.Input(shape=img_shape) # (H, W, C)
    y = keras.layers.Conv2D(
        32, kernel_size=3, strides=2, padding='same'
    )(img)
    y = keras.layers.LeakyReLU(alpha=0.2)(y)
    y = keras.layers.Dropout(0.25)(y)
    y = keras.layers.Conv2D(
        64, kernel_size=3, strides=2, padding='same'
    )(y)
    y = keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(y)
    y = keras.layers.BatchNormalization(momentum=0.8)(y)
    y = keras.layers.LeakyReLU(alpha=0.2)(y)
    y = keras.layers.Dropout(0.25)(y)
    y = keras.layers.Conv2D(
        128, kernel_size=3, strides=2, padding='same'
    )(y)
    y = keras.layers.BatchNormalization(momentum=0.8)(y)
    y = keras.layers.LeakyReLU(alpha=0.2)(y)
    y = keras.layers.Dropout(0.25)(y)
    y = keras.layers.Conv2D(
        256, kernel_size=3, strides=2, padding='same'
    )(y)
    y = keras.layers.BatchNormalization(momentum=0.8)(y)
    y = keras.layers.LeakyReLU(alpha=0.2)(y)
    y = keras.layers.Dropout(0.25)(y)
    y = keras.layers.Flatten()(y)
    y = keras.layers.Dense(1, activation='sigmoid')(y)

    return keras.Model(img, y, name='discriminator')

def get_mnist(data_dir=_DATADIR, batch_size=64):
    def norm_and_remove(img, label):
        """Normalization

        Args:
            img (tf.Tensor): mnist image tensor
            label (tf.float32): mnist image label
        """
        del label
        return (tf.cast(img, tf.float32) - 127.5) / 127.5
        # return tf.cast(img, tf.float32) / 255.0

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

            # Image generation
            image = self.model.generator.predict(z)

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

            # image = 255.0 * grid
            image = 255.0 * (grid + 1.) / 2.
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
        if epoch >= self.bound:
            weight = self.decay_rate ** (epoch / self.bound)
            d_lr.assign(self.lr * weight)
            g_lr.assign(self.lr * weight)

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

def model_build(argv):
    del argv
    model = build_discriminator((28, 28, 1))
    model.summary()

    model = build_generator(51)
    model.summary()

@main
def run(argv):
    del argv

    # Environment variable setting
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['NO_GCE_CHECK'] = 'true'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    logging.set_verbosity(logging.DEBUG)

    model = DCGAN()

    ds_train, _ = get_mnist(batch_size=model.batch_size)

    model.compile(
        run_eagerly=FLAGS.run_eagerly,
    )

    model.fit(
        ds_train,
        epochs=model.epochs,
        callbacks=model.callbacks,
    )
