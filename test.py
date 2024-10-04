import torch
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet

import transforms
from fcn.model import FCN16s
from fcn.test import test
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

    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    num_labels = len(dataset.class_to_idx)
    model = FCN16s(num_labels).to(DEVICE)
    model.load_state_dict(torch.load('./fcn/weights/fcn_vgg_16s_weights_dice_loss_oxford.pth'))

    for img, masks in test(model, data_loader, device=DEVICE):
        plot_with_masks(img, masks, num_labels)
