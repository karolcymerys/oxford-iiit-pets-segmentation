import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet
from tqdm import tqdm

import transforms
from loss_functions import IOULoss
from model import FCN32s

DEVICE = 'cuda:0'

if __name__ == '__main__':
    dataset = OxfordIIITPet(
        root='../data',
        transforms=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]),
        target_types='segmentation',
        split='test'
    )
    data_loader = DataLoader(dataset, batch_size=8)
    num_labels = len(dataset.class_to_idx)
    model = FCN32s(num_labels).to(DEVICE)
    model.load_state_dict(torch.load('./weights/fcn_vgg_32s_weights_25_v1_cross_entropy_loss_oxford.pth'))

    model = model.eval()
    loss_fn = IOULoss()
    ious = []

    with tqdm(data_loader, total=len(data_loader)) as samples:
        for batch_idx, batch in enumerate(samples, start=1):

            imgs = batch[0].to(DEVICE)
            targets = batch[1].to(DEVICE)
            outputs = model(imgs)

            loss = loss_fn(outputs, targets)
            ious.append(loss.item())

            samples.set_postfix({'Batch mIOU': loss.item()})

    print("mIOU: {:.4f}".format(float(np.mean(ious))))
