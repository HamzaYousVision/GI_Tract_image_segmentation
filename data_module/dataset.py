import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class GITrackDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((255, 255))]
        )

        self.images = glob.glob(f"{self.image_dir}/**/img.png", recursive=True)
        self.masks_large_bowel = glob.glob(
            f"{self.mask_dir}/**/large_bowel.png", recursive=True
        )
        self.masks_small_bowel = glob.glob(
            f"{self.mask_dir}/**/small_bowel.png", recursive=True
        )
        self.masks_stomach = glob.glob(
            f"{self.mask_dir}/**/stomach.png", recursive=True
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image = Image.open(self.images[index])
        maskmasks_large_bowel_lb_ = Image.open(self.masks_large_bowel[index])
        masks_large_bowel_ = Image.open(self.masks_small_bowel[index])
        masks_stomach_ = Image.open(self.masks_stomach[index])

        return {
            "image": self.transform(image),
            "mask_lb": self.transform(maskmasks_large_bowel_lb_),
            "mask_sb": self.transform(masks_large_bowel_),
            "mask_s": self.transform(masks_stomach_),
        }



