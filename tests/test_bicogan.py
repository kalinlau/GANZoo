"""Test: BiCoGAN."""
import os
import sys
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '../',
        )
    )
)

import pytest

class TestBiCoGAN:
    def test_path_config(self):
        from ganzoo.bicogan import _DATADIR, _WORKDIR

        data_relative_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                '../data'
            )
        )
        exps_relative_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                '../exps'
            )
        )

        assert data_relative_dir == _DATADIR
        assert exps_relative_dir == _WORKDIR

    def test_mnist_load(self):
        from ganzoo.bicogan import get_mnist
        bs = 4
        ds_train, _ = get_mnist(batch_size=bs)

        for img, label in ds_train.take(1):
            assert img.shape == (bs, 28, 28, 1)
            assert label.shape == (bs, 10)
            assert img.ndim == 4
            assert label.ndim == 2

    def test_model_io_dim(self):
        from ganzoo.bicogan import BiCoGAN

        model = BiCoGAN()

        img_shape = model.img_shape
        z_dim = model.z_dim
        n_dim = model.n_dim
        enc = model.encoder
        gen = model.generator
        dis = model.discriminator

        e_in = enc.get_layer('image').output
        e_out_z = enc.get_layer('out_z').output
        e_out_c = enc.get_layer('out_c').output
        g_in_z = gen.get_layer('intrinsic_z').output
        g_in_c = gen.get_layer('extrinsic_c').output
        g_out = gen.get_layer('out').output
        d_in_x = dis.get_layer('image').output
        d_in_z = dis.get_layer('intrinsic_z').output
        d_in_c = dis.get_layer('extrinsic_c').output
        d_out = dis.get_layer('out').output


        assert (
            e_in.shape == (None, img_shape) and
            e_out_z.shape == (None, z_dim) and
            e_out_c.shape == (None, n_dim)
        )

        assert (
            g_in_z.shape == (None, z_dim) and
            g_in_c.shape == (None, n_dim) and
            g_out.shape == (None, img_shape)
        )

        assert (
            d_in_z.shape == (None, z_dim) and
            d_in_c.shape == (None, n_dim) and
            d_in_x.shape == (None, img_shape) and
            d_out.shape == (None, 1)
        )

    def test_model_io_connection(self):
        from ganzoo.bicogan import BiCoGAN
        import tensorflow as tf
        import numpy as np

        model = BiCoGAN()

        # fake data
        img = tf.random.uniform(shape=(32, 784), minval=-1., maxval=1.)
        label = tf.one_hot(
            np.random.randint(0, 10, size=32),
            depth=10,
        )
        c = tf.random.uniform(shape=(32, 50), minval=-1., maxval=1.)

        enc = model.encoder
        gen = model.generator
        dis = model.discriminator

        c_, label_ = enc(img)
        img_ = gen([c, label])

        pred_real = dis([img, c_, label_])
        pred_fake = dis([img_, c, label])

        assert pred_real.numpy().ndim == 2 and pred_real.shape == (32, 1)
        assert pred_fake.numpy().ndim == 2 and pred_fake.shape == (32, 1)