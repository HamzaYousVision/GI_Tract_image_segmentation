import argparse
from data_module.data_generator import DataGenerator
from trainer import Trainer
from utils import parse_yaml_config


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
    # intialize data loaders




    # Initialize model




    # Initialize loss



    # Initialize optimizer




    # Initialize trainer




    # Train model




    
    if args.generate_data: 
        create_data()
    if args.train_data: 
        train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image segmentation main script")
    parser.add_argument("--generate_data", action="store_true", help="generate data")
    parser.add_argument("--train_data", action="store_true", help="train data")

    # Parse args from config yaml file
    config_args = parse_yaml_config("training.yaml")
    # Parse args from command line (override yaml file)
    config_args = parser.parse_args(namespace=config_args)

    main(config_args)
