import random
from typing import Tuple, List

import torchvision.transforms.functional as TF
from torchvision import transforms


class CustomTransforms:

    def __init__(self,
                 crop_size: Tuple[int, int] = (256, 256),
                 resize_size: List[int] = (128, 128),
                 rotation_angle: int = 25) -> None:
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.rotation_angle = rotation_angle

    def __call__(self, *args, **kwargs):
        objs = args

        # Random Crop
        if random.random() > 0.5:
            i, j, h, w = transforms.RandomCrop.get_params(
                objs[0], output_size=self.crop_size)
            objs = tuple([TF.crop(obj, i, j, h, w) for obj in objs])

        # Resize
        objs = tuple([TF.resize(obj, self.resize_size) for obj in objs])

        # Random horizontal flip
        if random.random() > 0.5:
            objs = tuple([TF.hflip(obj) for obj in objs])

        # Random rotation
        if random.random() > 0.5:
            objs = tuple([TF.rotate(obj, 25) for obj in objs])

        return objs

class CustomResizeTransforms:

    def __init__(self,
                 resize_size: List[int] = (128, 128)) -> None:
        self.resize_size = resize_size

    def __call__(self, *args, **kwargs):
        objs = args

        # Resize
        objs = tuple([TF.resize(obj, self.resize_size) for obj in objs])

        return objs

