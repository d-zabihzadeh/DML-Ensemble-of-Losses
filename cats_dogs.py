from os import path

from .base import *
import scipy.io

class CatsDogs(BaseDataset):
    def __init__(self, root, classes, transform = None):
        BaseDataset.__init__(self, root, classes, transform)
        annot = 'annotations'
        image_dir = path.join(root, 'images')
        self.im_paths, self.ys = [], []
        f = open(path.join(root, annot, 'list.txt'), encoding='utf8')
        index = 0
        for line in f:
            if (line.startswith('#')):
                continue
            values = line.split()
            y = int(values[1]) - 1
            if y in classes:
                self.im_paths.append(path.join(image_dir, values[0]+'.jpg'))
                self.ys.append(y)
                self.I += [index]
                index += 1





