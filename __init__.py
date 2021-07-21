from .cars import Cars
from .cub import CUBirds
from .sop import SOProducts
from .cats_dogs import CatsDogs
from . import utils
from .base import BaseDataset
from .flowers102 import Flowers102
import torch



_type = {
    'cars': Cars,
    'cub': CUBirds,
    'sop': SOProducts,
    'oxford_cats_dogs': CatsDogs,
    'flowers102': Flowers102
}


def load(name, root, classes, transform = None, img_size = 224):
    ds = _type[name](root = root, classes = classes, transform = transform)
    ds.name = name
    return ds
    

def load_ds(params, config):
    data_path = config['dataset'][params.ds]['root']
    if not hasattr(params, 'train_classes'):
        params.train_classes = config['dataset'][params.ds]['classes']['train']
    if not hasattr(params, 'test_classes'):
        params.test_classes = config['dataset'][params.ds]['classes']['eval']

    img_size = config['transform_parameters']['sz_crop']
    ds_train = load(params.ds, data_path, classes=params.train_classes, img_size=img_size,
                            transform=utils.make_transform(**config['transform_parameters']) )

    ds_test = load(params.ds, data_path, classes=params.test_classes,img_size=img_size,
                           transform=utils.make_transform(**config['transform_parameters'], is_train=False))

    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=params.batch_size, shuffle=True, drop_last=True,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=params.batch_size, shuffle=False, pin_memory=True)

    return train_loader, test_loader