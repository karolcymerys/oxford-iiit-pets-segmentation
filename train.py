from torch.utils.data import DataLoader, Subset
from torchvision.datasets import OxfordIIITPet

import transforms
from loss_functions import CrossEntropyLossWrapper, DiceLoss, BCELoss, DiceBCELoss, FocalLoss, TverskyLoss, FocalTverskyLoss
from fcn.model import FCN16s
from fcn.train import train

DEVICE = 'cuda:0'
BATCH_SIZE = 12

if __name__ == '__main__':
    dataset = OxfordIIITPet(
        root='./data',
        transforms=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor()
        ]),
        target_types='segmentation',
        split='trainval'
    )
    train_set = Subset(dataset, [idx for idx in range(len(dataset)) if idx % 10 != 0])
    validation_set = Subset(dataset, [idx for idx in range(len(dataset)) if idx % 10 == 0])

    train_data_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_data_loader = DataLoader(validation_set, batch_size=8)
    num_labels = len(dataset.class_to_idx)
    model = FCN16s(num_labels)
    model.init_weights()
    model = model.to(DEVICE)

    model = train(
        model,
        train_data_loader,
        validation_data_loader,
        DiceLoss(),
        'focal_loss_16s',
        device=DEVICE
    )
