from typing import Iterator, Tuple

import torch
from torch.utils.data import DataLoader

from dataset import PeopleClothingSegmentationDataset
from transforms import CustomTransforms
from utils import plot_with_masks


def extract_to_one(loader: DataLoader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    for batch_img, batch_masks in loader:
        for idx in range(batch_img.shape[0]):
            yield batch_img[idx, :, :, :].cpu(), batch_masks[idx, :, :].cpu()


if __name__ == '__main__':
    dataset = PeopleClothingSegmentationDataset(
        './dataset/png_images/IMAGES/',
        './dataset/png_masks/MASKS',
        './dataset/labels.csv',
        CustomTransforms(resize_size=(128, 128))
    )
    data_loader = DataLoader(dataset, batch_size=1)

    for img, masks in extract_to_one(data_loader):
        plot_with_masks(img, masks, len(dataset.idx2label))
