import argparse
from data_generator import DataGenerator
from trainer import Trainer

def create_data():
    root = "uw_medison"
    data_cleaner = DataGenerator(root)
    data_cleaner.transform_dataframe()
    data_cleaner.generate_images()
    data_cleaner.generate_masks()

def train():
    trainer = Trainer()
    trainer.train()

def main(args):
    if args.generate_data: 
        create_data()
    if args.train_data: 
        train()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image segmentation main script")
    parser.add_argument("--generate_data", action="store_true", help="generate data")
    parser.add_argument("--train_data", action="store_true", help="train data")
    args = parser.parse_args()

    main(args)