import os
import sys
import math

import numpy as np
from PIL import Image
import scipy.linalg

import chainer
import chainer.cuda
from chainer import Variable
from chainer import serializers
from chainer import cuda
import chainer.functions as F

sys.path.append(os.path.dirname(__file__))
sys.path.append('../')
from source.inception.inception_score import inception_score, Inception
from source.links.sn_convolution_2d import SNConvolution2D
from source.functions.max_sv import max_singular_value
from numpy.linalg import svd
from source.miscs.random_samples import sample_continuous, sample_categorical
import yaml
import source.yaml_utils as yaml_utils

from argparser import args
import langevin as sampler
import langevin_z as latent_sampler
from model_loader import dis
import source.inception.inception_score_tf as inception_score_tf

import time
import cupy as cp

def gen_images(gen, n=50000, batchsize=100):
    ims = []
    xp = gen.xp
    # start_time = time.time()
    # print('Start!')
    for i in range(0, n, batchsize):
        if i % 2500 == 2500-batchsize:
            print(str(i) + " generated")
        config = yaml_utils.Config(yaml.load(open(args.config_path)))
        is_conditional = config.updater['args']['conditional']
        if is_conditional:
            y = sample_categorical(gen.n_classes, batchsize, xp=gen.xp)
        else:
            y = None
        if args.sampling_space == 'pixel':
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                x = gen(batchsize, y=y)
            x = sampler.langevin(x, y, dis)
        elif args.sampling_space == 'latent':
            x, _ = latent_sampler.langevin(batchsize, gen, dis, y_fake=y, eval=True)
        x = chainer.cuda.to_cpu(x.data)
        x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        ims.append(x)
    ims = np.asarray(ims)
    _, _, _, h, w = ims.shape
    ims = ims.reshape((n, 3, h, w))
    # stop_time = time.time()
    # print('Stop! Time: '+str(stop_time-start_time))
    return ims


def gen_images_with_condition(gen, c=0, n=500, batchsize=100):
    ims = []
    xp = gen.xp
    for i in range(0, n, batchsize):
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            y = xp.asarray([c] * batchsize, dtype=xp.int32)
            x = gen(batchsize, y=y)
        x = chainer.cuda.to_cpu(x.data)
        x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        ims.append(x)
    ims = np.asarray(ims)
    _, _, _, h, w = ims.shape
    ims = ims.reshape((n, 3, h, w))
    return ims

def gen_eval_images(gen, n=50000, batchsize=100, seeds=1234, langevin_steps=5):
    '''
    langevin_steps: column
    '''
    ims = []
    xp = gen.xp
    xp.random.seed(seeds)
    for i in range(0, n, batchsize):
        print(i)
        config = yaml_utils.Config(yaml.load(open(args.config_path)))
        is_conditional = config.updater['args']['conditional']
        if is_conditional:
            y = sample_categorical(gen.n_classes, batchsize, xp=gen.xp)
        else:
            y = None
        if args.sampling_space == 'pixel':
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                x = gen(batchsize, y=y)
            for j in range(langevin_steps):
                x = sampler.langevin(x, y, dis)
                nx = chainer.cuda.to_cpu(x.data)
                nx = np.asarray(np.clip(nx * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
                ims.append(nx)
        elif args.sampling_space == 'latent':
            z = Variable(sample_continuous(gen.dim_z, batchsize, distribution=gen.distribution, xp=gen.xp))
            x = gen(batchsize, y=y, z=z)
            nx = chainer.cuda.to_cpu(x.data)
            nx = np.asarray(np.clip(nx * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
            ims.append(nx)
            for j in range(langevin_steps):
                x, z = latent_sampler.langevin(batchsize, gen, dis, y_fake=y, eval=True, given_z=z)
                nx = chainer.cuda.to_cpu(x.data)
                nx = np.asarray(np.clip(nx * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
                ims.append(nx)
    ims = list(map(list, zip(*ims)))
    ims = np.asarray(ims)
    _, _, _, h, w = ims.shape
    if args.sampling_space == 'latent':
        langevin_steps += 1
    ims = ims.reshape((n * langevin_steps, 3, h, w))
    return ims


def sample_generate_light(gen, dst, rows=5, cols=5, seed=0):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        x = gen_images(gen, n_images, batchsize=n_images)
        _, _, H, W = x.shape
        x = x.reshape((rows, cols, 3, H, W))
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((rows * H, cols * W, 3))
        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir + '/image_latest.png'
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(x).save(preview_path)

    return make_image


def sample_generate(gen, dst, rows=10, cols=10, seed=0):
    """Visualization of rows*cols images randomly generated by the generator."""
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        x = gen_images(gen, n_images, batchsize=n_images)
        _, _, h, w = x.shape
        x = x.reshape((rows, cols, 3, h, w))
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((rows * h, cols * w, 3))
        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir + '/image{:0>8}.png'.format(trainer.updater.iteration)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(x).save(preview_path)

    return make_image


def sample_generate_conditional(gen, dst, rows=10, cols=10, n_classes=1000, seed=0):
    """Visualization of rows*cols images randomly generated by the generator."""
    classes = np.asarray(np.arange(cols) * (n_classes / cols), dtype=np.int32)

    @chainer.training.make_extension()
    def make_image(trainer=None):
        np.random.seed(seed)
        xp = gen.xp
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = []
            for c in classes:
                x.append(gen_images_with_condition(gen, c=c, n=rows, batchsize=rows))
            x = np.concatenate(x, 0)
        _, _, h, w = x.shape
        x = x.reshape((rows, len(classes), 3, h, w))
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((rows * h, len(classes) * w, 3))

        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir + '/image{:0>8}.png'.format(
            trainer.updater.iteration if trainer is not None else None)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(x).save(preview_path)

    return make_image


def load_inception_model(path=None):
    path = path if path is not None else "%s/inception/inception_score.model" % os.path.dirname(__file__)
    model = Inception()
    serializers.load_hdf5(path, model)
    model.to_gpu()
    return model


def calc_inception(gen, batchsize=100, dst=None, path=None, n_ims=50000, splits=10):
    @chainer.training.make_extension()
    def evaluation(trainer=None):
        model = load_inception_model(path)
        ims = gen_images(gen, max(n_ims, batchsize), batchsize=batchsize).astype("f")
        mean, std = inception_score_tf.get_inception_score(ims, splits)
        # mean, std = inception_score(model, ims, splits=splits)
        chainer.reporter.report({
            'inception_mean': mean,
            'inception_std': std
        })
        if dst is not None:
            preview_dir = '{}/stats'.format(dst)
            preview_path = preview_dir + '/inception_score_{:0>8}.txt'.format(
                trainer.updater.iteration if trainer is not None else None)
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            np.savetxt(preview_path, np.array([mean, std]))

    return evaluation


def get_mean_cov(model, ims, batch_size=100):
    n, c, w, h = ims.shape
    n_batches = int(math.ceil(float(n) / float(batch_size)))
    xp = model.xp
    print('Batch size:', batch_size)
    print('Total number of images:', n)
    print('Total number of batches:', n_batches)
    ys = xp.empty((n, 2048), dtype=xp.float32)
    for i in range(n_batches):
        print('Running batch', i + 1, '/', n_batches, '...')
        batch_start = (i * batch_size)
        batch_end = min((i + 1) * batch_size, n)

        ims_batch = ims[batch_start:batch_end]
        ims_batch = xp.asarray(ims_batch)  # To GPU if using CuPy
        ims_batch = Variable(ims_batch)

        # Resize image to the shape expected by the inception module
        if (w, h) != (299, 299):
            ims_batch = F.resize_images(ims_batch, (299, 299))  # bilinear

        # Feed images to the inception module to get the features
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            y = model(ims_batch, get_feature=True)
        ys[batch_start:batch_end] = y.data

    mean = xp.mean(ys, axis=0).get()
    # cov = F.cross_covariance(ys, ys, reduce="no").datasets.get()
    cov = np.cov(ys.get().T)
    return mean, cov


def monitor_largest_singular_values(dis, dst):
    @chainer.training.make_extension()
    def evaluation(trainer=None):
        def _l2normalize(v, eps=1e-12):
            return v / (((v ** 2).sum()) ** 0.5 + eps)

        xp = dis.xp
        links = [[name, link] for name, link in sorted(dis.namedlinks())]
        sigmas = []
        for name, link in links:
            if isinstance(link, SNConvolution2D):
                W, u = link.W, link.u
                W_mat = W.reshape(W.shape[0], -1)
                sigma, _, _ = max_singular_value(W_mat, u)
                W_bar = cuda.to_cpu((W_mat.data / xp.squeeze(sigma.data)))
                _, s, _ = svd(W_bar)
                _sigma = s[0]
                print(name.strip('/'), _sigma)
                sigmas.append([name.strip('/'), _sigma])

        if dst is not None:
            preview_dir = '{}/sigmas'.format(dst)
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            preview_path = preview_dir + '/sigmas_{:0>8}.txt'.format(
                trainer.updater.iteration if trainer is not None else None)
            with open(preview_path, 'wb') as f:
                np.savetxt(f, np.array(sigmas, dtype=np.str), delimiter=" ", fmt="%s")

    return evaluation


def FID(m0, c0, m1, c1):
    ret = 0
    ret += np.sum((m0 - m1) ** 2)
    ret += np.trace(c0 + c1 - 2.0 * scipy.linalg.sqrtm(np.dot(c0, c1)))
    return np.real(ret)

def calc_inception_and_FID(gen, batchsize=200, stat_file="%s/cifar-10-fid.npz" % os.path.dirname(__file__), dst=None, path=None,
             n_ims=5000, splits=10):
    """Frechet Inception Distance proposed by https://arxiv.org/abs/1706.08500"""
    # http://bioinf.jku.at/research/ttur/)

    @chainer.training.make_extension()
    def evaluation(trainer=None):
        # loading models and generating new images
        # model = load_inception_model(path)
        ims = gen_images(gen, max(n_ims, batchsize), batchsize=batchsize).astype("f")
        stat = np.load(stat_file, allow_pickle=False)
        is_mean, is_std, fid_mean, fid_std = inception_score_tf.get_inception_and_FID(ims, splits, ref_stats=stat)

        # report and log IS
        chainer.reporter.report({
            'inception_mean': is_mean,
            'inception_std': is_std
        })
        if dst is not None:
            preview_dir = '{}/stats'.format(dst)
            preview_path = preview_dir + '/inception_score_{:0>8}.txt'.format(
                trainer.updater.iteration if trainer is not None else None)
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            np.savetxt(preview_path, np.array([is_mean, is_std]))

        # report and log FID
        chainer.reporter.report({
            'FID_mean': fid_mean,
            'FID_std': fid_std
        })
        if dst is not None:
            preview_dir = '{}/stats'.format(dst)
            preview_path = preview_dir + '/fid_{:0>8}.txt'.format(
                trainer.updater.iteration if trainer is not None else None)
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            np.savetxt(preview_path, np.array([fid_mean, fid_std]))

    return evaluation

def calc_FID(gen, batchsize=200, stat_file="%s/cifar-10-fid.npz" % os.path.dirname(__file__), dst=None, path=None,
             n_ims=5000, splits=10):
    """Frechet Inception Distance proposed by https://arxiv.org/abs/1706.08500"""
    # http://bioinf.jku.at/research/ttur/

    @chainer.training.make_extension()
    def evaluation(trainer=None):
        model = load_inception_model(path)
        stat = np.load(stat_file, allow_pickle=False)
        ims = gen_images(gen, max(n_ims, batchsize), batchsize=batchsize).astype("f")
        mean, std = inception_score_tf.get_fid(ims, ref_stats=stat, splits=splits)
        chainer.reporter.report({
            'FID_mean': mean,
            'FID_std': std
        })
        if dst is not None:
            preview_dir = '{}/stats'.format(dst)
            preview_path = preview_dir + '/fid_{:0>8}.txt'.format(
                trainer.updater.iteration if trainer is not None else None)
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            np.savetxt(preview_path, np.array([fid]))

    return evaluation
