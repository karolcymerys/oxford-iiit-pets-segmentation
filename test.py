import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from dataset import PeopleClothingSegmentationDataset
from segnet.modelv2 import SegNet
from segnet.test import test
from transforms import CustomTransforms
from utils import plot_with_masks

DEVICE = 'cuda:0'

if __name__ == '__main__':
    common_transforms = transforms.Compose([
        transforms.Resize((256, 256))
    ])

    dataset = PeopleClothingSegmentationDataset(
        './dataset/png_images/IMAGES/',
        './dataset/png_masks/MASKS',
        './dataset/labels.csv',
        CustomTransforms(resize_size=(256, 256))
    )
    test_set = Subset(dataset, [idx for idx in range(len(dataset)) if idx % 10 == 0])

    data_loader = DataLoader(test_set, batch_size=16)
    num_labels = len(dataset.idx2label)
    model = SegNet(num_labels).to(DEVICE)
    model.load_state_dict(torch.load('segnet/weights/seg_net_weights_5_cross_entropy.pth'))

    for img, masks in test(model, data_loader, common_transforms, device=DEVICE):
        plot_with_masks(img, masks, num_labels)
