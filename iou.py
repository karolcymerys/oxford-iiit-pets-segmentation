import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from torchvision import transforms as torch_transforms

import transforms
from dataset import PeopleClothingSegmentationDataset
from loss_functions import IOULoss
from segnet.modelv2 import SegNet

DEVICE = 'cuda:0'

if __name__ == '__main__':
    common_transforms = torch_transforms.Compose([
        torch_transforms.Resize((224, 224))
    ])

    dataset = PeopleClothingSegmentationDataset(
        './dataset/png_images/IMAGES/',
        './dataset/png_masks/MASKS',
        './dataset/labels.csv',
        transforms.CustomTransforms(resize_size=(224, 224))
    )
    test_set = Subset(dataset, [idx for idx in range(len(dataset)) if idx % 10 == 0])
    data_loader = DataLoader(test_set, batch_size=8)
    num_labels = len(dataset.idx2label)
    model = SegNet(num_labels).to(DEVICE)
    # model.load_state_dict(torch.load('./segnet/weights/seg_net_v2_weights_20_dice_loss.pth'))

    model = model.eval()
    loss_fn = IOULoss()
    ious = []

    for batch_idx, batch in enumerate(data_loader, start=1):
        imgs = batch[0].to(DEVICE)
        targets = batch[1].to(DEVICE)
        outputs = model(common_transforms(imgs))

        loss = loss_fn(outputs, common_transforms(targets))
        ious.append(loss.item())

    print("mIOU: {:.4f}".format(float(np.mean(ious))))
