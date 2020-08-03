import os, sys
import numpy as np
import argparse
import chainer

base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../'))
from evaluation import gen_images
import yaml
import source.yaml_utils as yaml_utils

from argparser import args
# Model
from model_loader import gen, dis

import source.inception.inception_score_tf as inception_score_tf
from evaluation import load_inception_model
from source.inception.inception_score import inception_score, Inception


# def load_models(config):
#     gen_conf = config.models['generator']
#     gen = yaml_utils.load_model(gen_conf['fn'], gen_conf['name'], gen_conf['args'])
#     return gen


def main():
    chainer.cuda.get_device_from_id(args.gpu).use()

    np.random.seed(1234)
    xp = gen.xp
    n = int(5000 * args.splits)
    #for _ in range(50):
    #     gen(128) 
    print("Gen")
    ims = gen_images(gen, n, batchsize=125).astype("f")
    print(np.max(ims), np.min(ims))

    if args.tf:
        # mean, std = inception_score.get_inception_score(ims, args.splits)
        stat = np.load(args.FID_stat_file, allow_pickle=False)
        is_mean, is_std, fid_mean, fid_std = inception_score_tf.get_inception_and_FID(ims, args.splits, ref_stats=stat)
        print(is_mean, is_std, fid_mean, fid_std)
    else:
        model = load_inception_model(args.inception_model_path)
        mean, std = inception_score(model, ims, splits=args.splits)
        print(mean, std)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    np.savetxt('{}/inception_score.txt'.format(args.results_dir),
               np.array([is_mean, is_std]))
    np.savetxt('{}/FID.txt'.format(args.results_dir),
               np.array([fid_mean, fid_std]))


if __name__ == '__main__':
    main()
