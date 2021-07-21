from os import path, listdir

from .base import *
import scipy.io

class Flowers102(BaseDataset):
    def __init__(self, root, classes, transform = None):
        BaseDataset.__init__(self, root, classes, transform)
        image_dir = path.join(root, 'train')
        self.im_paths, self.ys = [], []
        sub_folders = listdir(image_dir)
        index = 0
        for d in sub_folders:
            y = int(d) - 1
            if (y in classes):
                fnames = [path.join(image_dir, d, file) for file in listdir(path.join(image_dir, d))]
                self.im_paths += fnames
                m = len(fnames)
                self.ys += ([y] * m)
                self.I += list(range(index, index + m))
                index += m
