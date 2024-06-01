import os
from csv import DictReader
from typing import Dict, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PeopleClothingSegmentationDataset(Dataset):
    def __init__(self,
                 images_path: str,
                 masks_path: str,
                 labels_filepath: str,
                 custom_transforms: Callable = lambda inp: inp) -> None:
        self.idx2label = self.__load_labels(labels_filepath)
        self.dataset = self.__load_dataset(images_path, masks_path)
        self.transforms = custom_transforms

    @staticmethod
    def __load_dataset(images_path: str, masks_path: str) -> Dict[int, Dict[str, str]]:
        dataset = list()
        for img_filename in list(sorted(os.listdir(images_path))):
            idx = img_filename.split('_')[1].split('.')[0]
            seg_filename = f'seg_{idx}.png'
            if os.path.isfile(os.path.join(masks_path, seg_filename)):
                dataset.append({
                    'img': os.path.join(images_path, img_filename),
                    'mask': os.path.join(masks_path, seg_filename),
                })

        return dict(enumerate(dataset))

    def __getitem__(self, item: int) -> [torch.Tensor, torch.Tensor]:
        item = self.dataset[item]

        to_tensor = transforms.ToTensor()
        pil_to_tensor = transforms.PILToTensor()

        img = to_tensor(Image.open(item['img']))
        mask = pil_to_tensor(Image.open(item['mask']))

        return self.transforms(img, mask)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def __load_labels(filepath: str) -> Dict[int, str]:
        with open(filepath, 'r') as file:
            csv_reader = DictReader(file, fieldnames=['id', 'label'])
            next(csv_reader)
            return {int(line['id']): line['label'] for line in csv_reader}
