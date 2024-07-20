import torch
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet

import transforms
from segnet.model import SegNet
from segnet.test import test
from utils import plot_with_masks

DEVICE = 'cuda:0'

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

    data_loader = DataLoader(dataset, batch_size=16)
    num_labels = len(dataset.class_to_idx)
    model = SegNet(num_labels).to(DEVICE)
    model.load_state_dict(torch.load('./segnet/weights/seg_net_weights_36_v1_dice_loss_oxford.pth'))

    for img, masks in test(model, data_loader, device=DEVICE):
        plot_with_masks(img, masks, num_labels)
