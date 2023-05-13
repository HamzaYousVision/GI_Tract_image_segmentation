from data_generator import DataGenerator

def main():
    root = "uw_medison"
    data_cleaner = DataGenerator(root)
    data_cleaner.transform_dataframe()
    data_cleaner.generate_images()
    data_cleaner.generate_masks()


if __name__ == "__main__":
    main()
