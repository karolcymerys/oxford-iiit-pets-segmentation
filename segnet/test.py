import os
from typing import Tuple, Iterator

import torch
from torch import log_softmax
from torch.utils.data import DataLoader
from torchvision import transforms

from loss_functions import IOULoss
from segnet.model import SegNet

dir_path = os.path.dirname(os.path.realpath(__file__))


def test(model: SegNet,
         data_loader: DataLoader,
         device: str = 'cpu') -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    model = model.eval()

    loss_fn = IOULoss()
    ious = []
    for batch_idx, batch in enumerate(data_loader, start=1):
        imgs = batch[0].to(device)
        targets = batch[1].to(device)
        outputs = model(imgs)
        masks = log_softmax(outputs, dim=1).exp().argmax(dim=1).unsqueeze(dim=1)

        loss = loss_fn(outputs, targets)
        ious.append(loss.item())

        resize = transforms.Resize(imgs.shape[2:])

        yield imgs.cpu(), resize(masks).cpu()
