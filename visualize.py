from torch.utils.data import DataLoader

from torchvision.datasets import OxfordIIITPet

import transforms
from utils import plot_with_masks

BATCH_SIZE = 16

if __name__ == '__main__':
    dataset = OxfordIIITPet(
        root='./data',
        transforms=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]),
        target_types='segmentation',
        split='test'
    )
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    for batch_imgs, batch_masks in data_loader:
        plot_with_masks(batch_imgs, batch_masks, len(dataset.class_to_idx))
