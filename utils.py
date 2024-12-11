from torch.utils.data import Dataset 
import os
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.utils.data import WeightedRandomSampler

class CheXpertDataset(Dataset):
    def __init__(self, data, root_dir, mode='train', transforms=None):
        self.data = data.to_numpy()
        self.labels = torch.tensor(data.values)
        self.root_dir = root_dir
        self.img_paths = [os.path.join(root_dir, img_path) for img_path in data.index]
        self.transforms = transforms.get(mode)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transforms:
            image = self.transforms(image=np.array(image))['image']

        return (image, label)
    
def get_weighted_random_sampler(data):
    weights = 1/data.sum()
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler

transforms = {
    'train': A.Compose([
        A.Resize(224, 224),
        A.Affine(scale=(0.9, 1.1), p=0.5),
        A.OneOf([A.Affine(rotate=(-20, 20), p=0.5), A.Affine(shear=(-5, 5), p=0.5)], p=0.5),
        A.Affine(translate_percent=(-0.05, 0.05), p=0.5),
        A.Normalize(mean[mode], std[mode]),
        ToTensorV2()
    ]),
    'val': A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean[mode], std[mode]),
        ToTensorV2()
    ]),
}