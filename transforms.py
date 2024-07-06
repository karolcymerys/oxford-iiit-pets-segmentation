import random
from typing import Tuple, List

import torchvision.transforms.functional as TF
from torchvision import transforms


class CustomTransforms:

    def __init__(self,
                 crop_size: Tuple[int, int] | None = None,
                 resize_size: Tuple[int, int] | None = None,
                 rotation_angle: int | None = None,
                 horizontal_flip: bool = False,
                 brightness_adjustment: bool = False,
                 sharpness_adjustment: bool = False) -> None:
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.rotation_angle = rotation_angle
        self.horizontal_flip = horizontal_flip
        self.brightness_adjustment = brightness_adjustment
        self.sharpness_adjustment = sharpness_adjustment

    def __call__(self, *args, **kwargs):
        objs = args

        # Random Crop
        if self.crop_size and random.random() > 0.5:
            i, j, h, w = transforms.RandomCrop.get_params(
                objs[0], output_size=self.crop_size)
            objs = tuple([TF.crop(obj, i, j, h, w) for obj in objs])

        # Resize
        if self.resize_size:
            objs = tuple([TF.resize(obj, list(self.resize_size)) for obj in objs])

        # Random horizontal flip
        if self.horizontal_flip and random.random() > 0.5:
            objs = tuple([TF.hflip(obj) for obj in objs])

        # Random rotation
        if self.rotation_angle and random.random() > 0.5:
            factor = random.randrange(-self.rotation_angle, self.rotation_angle)
            objs = tuple([TF.rotate(obj, factor) for obj in objs])

        # Random brightness adjustment
        if self.brightness_adjustment and random.random() > 0.5:
            factor = random.randrange(0, 3)
            objs = tuple([TF.adjust_brightness(objs[0], factor), *objs[1:]])

        # Random sharpness adjustment
        if self.sharpness_adjustment and random.random() > 0.5:
            factor = random.randrange(0, 3)
            objs = tuple([TF.adjust_sharpness(objs[0], factor), *objs[1:]])

        # Random contrast adjustment
        #if random.random() > 0.5:
        #    factor = random.randrange(0, 3)
        #    objs = tuple([TF.adjust_contrast(objs[0], factor), *objs[1:]])

        # Random saturation adjustment
        #if random.random() > 0.5:
        #    factor = random.randrange(0, 3)
        #    objs = tuple([TF.adjust_saturation(objs[0], factor), *objs[1:]])

        return objs
