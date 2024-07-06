from torch.utils.data import DataLoader

from dataset import PeopleClothingSegmentationDataset
from transforms import CustomTransforms
from utils import plot_with_masks

BATCH_SIZE = 16

if __name__ == '__main__':
    dataset = PeopleClothingSegmentationDataset(
        './dataset/png_images/IMAGES/',
        './dataset/png_masks/MASKS',
        './dataset/labels.csv',
        CustomTransforms(resize_size=(128, 128))
    )
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    for batch_imgs, batch_masks in data_loader:
        plot_with_masks(batch_imgs, batch_masks, len(dataset.idx2label))
