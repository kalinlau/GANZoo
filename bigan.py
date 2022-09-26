""" BiGAN (https://arxiv.org/pdf/1605.09782.pdf)"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['NO_GCE_CHECK'] = 'true'

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


class BiGAN(Model):
    def __init__(self, name='BiGAN', **kwargs):
        super().__init__(name=name, **kwargs)
        # Image size
        self.H, self.W, self.C = 28, 28, 1
        self.image_shape = self.H * self.W * self.C
        # Latent dim
        self.z_dim = 50
        # Batch size
        self.batch_size = 128
        # Epochs
        self.epochs = 400
        # Init learning rate
        self.lr = 1e-4
        timeshift = datetime.today().strftime('%Y%m%d-%H:%M:%S')
        self.workdir = 'exps/' + f'BiGAN/f{timeshift}'
        self.logdir = os.path.join(self.workdir, 'logs')
        self.imgdir = os.path.join(self.workdir, 'imgs/')

        tf.io.gfile.makedirs(self.workdir)
        tf.io.gfile.makedirs(self.logdir)
        tf.io.gfile.makedirs(self.imgdir)

        self.generator = build_generator(self.image_shape, self.z_dim)
        self.encoder = build_encoder(self.image_shape, self.z_dim)
        self.discriminator = build_discriminator(self.image_shape, self.z_dim)

        gjson = self.generator.to_json()
        ejson = self.encoder.to_json()
        djson = self.discriminator.to_json()

        with open(os.path.join(self.workdir, 'model.json'), 'w') as json_file:
            json_file.write(gjson)
            json_file.write(ejson)
            json_file.write(djson)

        self.callbacks = [
            SaveImage(workdir=self.imgdir, latent_dim=self.z_dim),
            LearningRateDecay(init_lr=self.lr, decay_steps=self.epochs / 2.0),
            keras.callbacks.TensorBoard(log_dir=self.logdir),
        ]

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.d_optimizer = Adam(learning_rate=self.lr, beta_1=0.5, beta_2=0.999)
        self.g_optimizer = Adam(learning_rate=self.lr, beta_1=0.5, beta_2=0.999)
        self.e_optimizer = Adam(learning_rate=self.lr, beta_1=0.5, beta_2=0.999)
        self.d_loss_metric = keras.metrics.Mean(name='d_loss')
        self.g_loss_metric = keras.metrics.Mean(name='g_loss')
        self.lr_metric = Monitor(name='d_lr')

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric, self.lr_metric]

    def train_step(self, data):
        image = data
        batch_size = tf.shape(image)[0]
        img = tf.reshape(image, shape=(batch_size, -1), name='flatten_img')
        z = tf.random.uniform(shape=(batch_size, self.z_dim), minval=-1, maxval=1., dtype=tf.float32)

        img_ = self.generator(z)
        z_ = self.encoder(img)

        d_inputs = [tf.concat([img_, img], axis=0),
                    tf.concat([z, z_], axis=0)]
        d_preds = self.discriminator(d_inputs)
        pred_g, pred_e = tf.split(d_preds,num_or_size_splits=2, axis=0)

        d_loss = tf.reduce_mean(tf.math.softplus(pred_g)) + \
            tf.reduce_mean(tf.math.softplus(-pred_e))
        g_loss = tf.reduce_mean(tf.math.softplus(-pred_g))
        e_loss = tf.reduce_mean(tf.math.softplus(pred_e))

        d_gradients = tf.gradients(d_loss, self.discriminator.trainable_variables)
        g_gradients = tf.gradients(g_loss, self.generator.trainable_variables)
        e_gradients = tf.gradients(e_loss, self.encoder.trainable_variables)

        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        self.e_optimizer.apply_gradients(zip(e_gradients, self.encoder.trainable_variables))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        self.lr_metric.update_state(self.d_optimizer.learning_rate)

        return {'d_loss': self.d_loss_metric.result(),
                'g_loss': self.g_loss_metric.result(),
                'd_lr':self.lr_metric.result(),}

class SaveImage(keras.callbacks.Callback):
    def __init__(self, workdir='.', interval=10, num_img=100, latent_dim=50):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.workdir = workdir
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if not tf.math.mod(epoch, self.interval):
            z = tf.random.uniform(shape=(self.num_img, self.latent_dim), minval=-1., maxval=1.)
            image = self.model.generator.predict(z)
            image = np.reshape(image, (10, 10, 28, 28))
            image = np.transpose(image, (0, 2, 1, 3))
            image = np.reshape(image, (10 * 28, 10 * 28))
            image = 255 * (image + 1) / 2
            image = np.clip(image,0,255)
            image = image.astype("uint8")
            tf.io.gfile.makedirs(self.workdir)
            Image.fromarray(image, "L").save(
                os.path.join(self.workdir, "G_z-{}.png".format(epoch))
            )

class LearningRateDecay(keras.callbacks.Callback):
    def __init__(self, init_lr, decay_steps):
        self.lr = init_lr
        self.decay_rate = 0.1
        self.bound = decay_steps

    def on_epoch_begin(self, epoch, logs=None):
        d_lr = self.model.d_optimizer.learning_rate
        g_lr = self.model.g_optimizer.learning_rate
        e_lr = self.model.e_optimizer.learning_rate
        if epoch >= self.bound:
            weight = self.decay_rate ** (epoch / self.bound)
            d_lr.assign(self.lr * weight)
            g_lr.assign(self.lr * weight)
            e_lr.assign(self.lr * weight)

class Monitor(keras.metrics.Metric):
    def __init__(self, name='lr_monitor', **kwargs):
        super(Monitor, self).__init__(name=name, **kwargs)
        self.lr = tf.Variable(0.0, trainable=False)

    def update_state(self, lr):
        self.lr.assign(lr)

    def result(self):
        return self.lr

    def reset_state(self):
        self.lr.assign(0.0)

def build_generator(
        img_shape,
        z_dim,
        reg=lambda: L1L2(l1=0., l2=2.5e-5),
    ):
    """Generator Network

    Architecture: (1024)FC_ReLU-(1024)FC_ReLU_BN-(784)FC
    Regularization: L2(2.5e-5)
    Kernal Initialization: Normal(mean=0., stddev=0.02)
    """
    x = Input(z_dim)
    y = Dense(
        1024,
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
        kernel_regularizer=reg(),
    )(x)
    y = ReLU()(y)
    y = Dense(
        1024,
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
        kernel_regularizer=reg(),
    )(y)
    y = ReLU()(y)
    y = BatchNormalization()(y)
    y = Dense(
        img_shape,
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
        kernel_regularizer=reg(),
    )(y)
    return Model(x, y, name='generator')

def build_encoder(
        img_shape,
        z_dim,
        reg=lambda: L1L2(l1=0., l2=2.5e-5),
    ):
    """Encoder Network

    Architecture: (1024)FC_lrelu-(1024)FC_lrelu_BN-(z_dim)FC
    Regularization: L2(2.5e-5)
    Kernal Initialization: Normal(mean=0., stddev=0.02)
    """
    # ----- Original Arch -----
    x = Input(img_shape)
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
    y = Dense(
        z_dim,
        kernel_regularizer=reg(),
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)
    )(y)

    return Model(x, y, name='encoder')

def build_discriminator(
        img_shape,
        z_dim,
        reg=lambda: L1L2(l1=0, l2=2e-5),
    ):
    """Discriminator Network

    Architecture: (1024)FC_lrelu-(1024)FC_lrelu_BN-(z_dim)FC
    Regularization: L2(2.5e-5)
    Kernal Initialization:
        Normal(mean=0.0, stddev=0.5), Normal(mean=0., stddev=0.02)
    """
    x = Input(img_shape)
    z = Input(z_dim)
    y = Concatenate()([x,z])
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
    )(y)
    return Model([x, z], [y], name='discriminator')

def get_mnist():
    """TODO: Doubt there is a bug on shuffling, but model.fit()
    should do it by default."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = np.reshape(x_train, (-1, 28*28*1))
    x_train = (x_train.astype("float32") / 255) * 2 - 1

    return x_train

def get_tfds_mnist(data_dir='./data', batch_size=128):
    def norm_and_remove(img, label):
        """Normalize to [-1, 1] and Remove label

        Args:
            img (tf.Tensor): mnist image tensor
            label (tf.float32): mnist image label

        Returns:
            img: normalized image
        """
        return (tf.cast(img, tf.float32) - 127.5) / 127.5

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

if __name__ == '__main__':
    model = BiGAN()

    x_train, _ = get_tfds_mnist(batch_size=model.batch_size)

    model.compile()

    model.fit(
        x_train,
        epochs=model.epochs,
        callbacks=model.callbacks,
    )
