from torch.utils.data import DataLoader, Subset

from dataset import PeopleClothingSegmentationDataset
from loss_functions import CrossEntropyLossWrapper, DiceLoss
from segnet.modelv2 import SegNet
from segnet.train import train
from transforms import CustomTransforms

DEVICE = 'cuda:0'
BATCH_SIZE = 24

if __name__ == '__main__':
    dataset = PeopleClothingSegmentationDataset(
        './dataset/png_images/IMAGES/',
        './dataset/png_masks/MASKS',
        './dataset/labels.csv',
        CustomTransforms(
            crop_size=(256, 256),
            resize_size=(224, 224),
            rotation_angle=15,
            horizontal_flip=True,
            brightness_adjustment=True,
            sharpness_adjustment=True
        )
    )
    train_set = Subset(dataset, [idx for idx in range(len(dataset)) if idx % 10 != 0])
    validation_set = Subset(dataset, [idx for idx in range(len(dataset)) if idx % 10 == 0])

    train_data_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_data_loader = DataLoader(validation_set, batch_size=8)
    num_labels = len(dataset.idx2label)
    model = SegNet(num_labels)
    model.init_weights()
    model = model.to(DEVICE)

    model = train(
        model,
        train_data_loader,
        validation_data_loader,
        DiceLoss(), # CustomCrossEntropyLoss()
        device=DEVICE
    )
