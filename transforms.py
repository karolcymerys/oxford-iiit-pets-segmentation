import random
from typing import Tuple, Callable, List

import torchvision.transforms.functional as TF
from torchvision import transforms


class Compose:
    def __init__(self, ts: List[Callable]) -> None:
        self.ts = ts

    def __call__(self, *args, **kwargs):
        params = args
        for t in self.ts:
            params = t(*params)
        return params


class Resize:

    def __init__(self, target_size: Tuple[int, int]) -> None:
        self.target_size = target_size

    def __call__(self, *args, **kwargs):
        return tuple([TF.resize(obj, list(self.target_size)) for obj in args])


class RandomCrop:

    def __init__(self, target_size: Tuple[int, int], threshold: float = 0.5) -> None:
        self.target_size = target_size
        self.threshold = threshold

    def __call__(self, *args, **kwargs):
        objs = args
        if random.random() > self.threshold:
            i, j, h, w = transforms.RandomCrop.get_params(args[0], output_size=self.target_size)
            objs = tuple([TF.crop(obj, i, j, h, w) for obj in objs])
        return objs


class RandomRotation:

    def __init__(self, angle: int, threshold: float = 0.5) -> None:
        self.angle = angle
        self.threshold = threshold

    def __call__(self, *args, **kwargs):
        objs = args
        if random.random() > self.threshold:
            angle = random.randrange(-self.angle, self.angle)
            objs = tuple([TF.rotate(obj, angle) for obj in objs])
        return objs


class RandomHorizontalFlip:

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def __call__(self, *args, **kwargs):
        objs = args
        if random.random() > self.threshold:
            objs = tuple([TF.hflip(obj) for obj in objs])
        return objs


class ToTensor:

    def __init__(self, target_with_normalization: bool = False) -> None:
        self.target_with_normalization = target_with_normalization
        self.to_tensor = transforms.ToTensor()
        self.pil_to_tensor = transforms.PILToTensor()

    def __call__(self, *args, **kwargs):
        return tuple([self.pil_to_tensor(obj) if not self.target_with_normalization and idx > 0
                      else self.to_tensor(obj)
                      for idx, obj in enumerate(args)])


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

        return objs
