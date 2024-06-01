from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import PeopleClothingSegmentationDataset
from segnet.model import SegNet
from segnet.train import train

DEVICE = 'cuda:0'

if __name__ == '__main__':
    common_transforms = transforms.Compose([
        transforms.Resize((128, 128))
    ])

    dataset = PeopleClothingSegmentationDataset(
        './dataset/png_images/IMAGES/',
        './dataset/png_masks/MASKS',
        './dataset/labels.csv',
        common_transforms

    )
    dataloader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=True)
    num_labels = len(dataset.idx2label)
    model = SegNet(num_labels)
    model.init_weights()
    model = model.to(DEVICE)

    model = train(model, dataloader, num_labels, device=DEVICE)
