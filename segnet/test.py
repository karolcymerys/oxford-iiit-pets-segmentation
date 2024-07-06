import os
from typing import Tuple, Iterator, Callable

import torch
from torch import softmax
from torch.utils.data import DataLoader
from torchvision import transforms

from segnet.model import SegNet

dir_path = os.path.dirname(os.path.realpath(__file__))


def test(model: SegNet,
         data_loader: DataLoader,
         custom_transforms: Callable = lambda inp: inp,
         device: str = 'cpu') -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    model = model.eval()
    for batch_idx, batch in enumerate(data_loader, start=1):
        imgs = batch[0].to(device)

        outputs = model(custom_transforms(imgs))
        masks = softmax(outputs, dim=1).argmax(dim=1).unsqueeze(dim=1)

        resize = transforms.Resize(imgs.shape[2:])

        yield imgs.cpu(), resize(masks).cpu()
