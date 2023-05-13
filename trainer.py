from torch.utils.data import DataLoader, random_split
from dataset import GITrackDataset


class Trainer:
    def __init__(self):
        self.epochs = 10
        self.define_optimizer()
        self.create_dataset()
        self.create_loaders()

    def define_optimizer(self):
        pass

    def create_dataset(self):
        datset = GITrackDataset("uw_medison/data/images", "uw_medison/data/masks")
        train_len = len(datset) // 10
        val_len = len(datset) - train_len

        self.train_dataset, self.valid_dataset = random_split(
            datset, [train_len, val_len]
        )

    def create_loaders(self):
        self.train_dataset = DataLoader(self.train_dataset, batch_size=8)
        self.valid_dataset = DataLoader(self.train_dataset, batch_size=8)

    def train(self):
        for data in self.train_dataset:
            print(data["image"].shape, data["mask_lb"].shape)
