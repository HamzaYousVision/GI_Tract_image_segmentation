import os 
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset

class GITrackDataset(Dataset): 
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = glob.glob(f"{self.image_dir}/**/img.png", recursive=True)
        self.masks_large_bowel = glob.glob(f"{self.mask_dir}/**/large_bowel.png", recursive=True)
        self.masks_small_bowel = glob.glob(f"{self.mask_dir}/**/small_bowel.png", recursive=True)
        self.masks_stomach = glob.glob(f"{self.mask_dir}/**/mask_stomach.png", recursive=True)


    def __len__(self): 
        return len(self.images)

    def __getitem__(self, index):
        if torch.is_tensor(index): 
            index = index.tolist() 

        image = Image.open(self.images[index])
        mask_lb = Image.open(self.masks_large_bowel[index])
        mask_sb = Image.open(self.masks_small_bowel[index])
        mask_s = Image.open(self.masks_stomach[index])
        sample = {"image": image, 'mask_lb': mask_lb, "mask_sb": mask_sb, "mask_s": mask_s}

        if self.transform is not None: 
            sample = self.transform(sample)
        return sample


def main(): 
    git = GITrackDataset("uw_medison/data/images", "uw_medison/data/masks")
    print(len(git))


if __name__ == "__main__": 
    main()