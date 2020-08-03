import chainer
import numpy as np
from chainer.dataset import dataset_mixin


class STL10Dataset(dataset_mixin.DatasetMixin):
    def __init__(self, test=False):
        from argparser import args
        images = np.load(args.data_dir)
        labels = np.zeros([images.shape[0], 1])  # no use,
        self.dset = (images, labels)
        print("load stl-10.  shape: ", len(self.dset[0]))

    def __len__(self):
        return len(self.dset[0])

    def get_example(self, i):
        image = np.asarray(self.dset[0][i] / 128. - 1., np.float32)
        image += np.random.uniform(size=image.shape, low=0., high=1. / 128)
        return image, self.dset[1][i]
