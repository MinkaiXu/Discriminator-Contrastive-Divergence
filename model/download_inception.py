# Code derived from https://github.com/openai/improved-gan/tree/master/inception_score
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
import scipy.misc
import math
import sys
import chainer
from chainer import functions as F

MODEL_DIR = '/mnt/cephfs_new_wj/bytetrans/xuminkai/DCD/model/inception'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

# if not os.path.exists(MODEL_DIR):
#     os.makedirs(MODEL_DIR)
filename = DATA_URL.split('/')[-1]
filepath = os.path.join(MODEL_DIR, filename)
# if not os.path.exists(filepath):
#     def _progress(count, block_size, total_size):
#         sys.stdout.write('\r>> Downloading %s %.1f%%' % (
#             filename, float(count * block_size) / float(total_size) * 100.0))
#         sys.stdout.flush()
#     filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
#     print()
#     statinfo = os.stat(filepath)
#     print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)