import glob
import os
import cv2
import pandas as pd
import numpy as np
import shutil

import utils


class DataCleaner:
    def __init__(self, root):
        self.root = root
        self.image_folder = os.path.join(root, "train", "*")
        self.df = pd.read_csv(os.path.join(root, "train.csv"))

    def add_case_day_slice(self):
        self.df["case"] = self.df["id"].apply(
            lambda x: int(x.split("_")[0].replace("case", ""))
        )
        self.df["day"] = self.df["id"].apply(
            lambda x: int(x.split("_")[1].replace("day", ""))
        )
        self.df["slice"] = self.df["id"].apply(lambda x: x.split("_")[3])

    def add_image_path(self):
        all_images = glob.glob(os.path.join(self.root, "*", "*", "*", "*", "*"))
        x = all_images[0].rsplit("/", 4)[0]

        path_partial_list = []
        for i in range(0, self.df.shape[0]):
            path_partial_list.append(
                os.path.join(
                    x,
                    f"case{self.df['case'].values[i]}",
                    f"case{self.df['case'].values[i]}_day{self.df['day'].values[i]}",
                    "scans",
                    f"slice_{self.df['slice'].values[i]}",
                )
            )
        self.df["path_partial"] = path_partial_list

        path_partial_list = []
        for i in range(0, len(all_images)):
            path_partial_list.append(str(all_images[i].rsplit("_", 4)[0]))

        tmp_df = pd.DataFrame()
        tmp_df["path_partial"] = path_partial_list
        tmp_df["path"] = all_images

        self.df = self.df.merge(tmp_df, on="path_partial").drop(
            columns=["path_partial"]
        )
        self.df["width"] = self.df["path"].apply(
            lambda x: int(x[:-4].rsplit("_", 4)[1])
        )
        self.df["height"] = self.df["path"].apply(
            lambda x: int(x[:-4].rsplit("_", 4)[2])
        )

    def create_transformed_dataframe(self):
        self.df_clean = pd.DataFrame({"id": self.df["id"][::3]})

        self.df_clean["large_bowel"] = self.df["segmentation"][::3].values
        self.df_clean["small_bowel"] = self.df["segmentation"][1::3].values
        self.df_clean["stomach"] = self.df["segmentation"][2::3].values

        self.df_clean["path"] = self.df["path"][::3].values
        self.df_clean["case"] = self.df["case"][::3].values
        self.df_clean["day"] = self.df["day"][::3].values
        self.df_clean["slice"] = self.df["slice"][::3].values
        self.df_clean["width"] = self.df["width"][::3].values
        self.df_clean["height"] = self.df["height"][::3].values

        self.df_clean = self.df_clean.reset_index(drop=True)
        self.df_clean = self.df_clean.fillna("")

    def transform_dataframe(self):
        self.add_case_day_slice()
        self.add_image_path()
        self.create_transformed_dataframe()
    
    def generate_masks(self):
        for _, row in self.df_clean.iterrows():
            case = str(row["case"])
            day = str(row["day"])
            slice = str(row['slice'])

            mask_lb = row["large_bowel"]
            mask_sb = row["small_bowel"]
            mask_s = row["stomach"]

            h = row["height"]
            w = row["width"]

            path = os.path.join(self.root, f"data/masks/{case}/{day}/{slice}")
            if (not os.path.exists(path)):
                os.makedirs(path)

            img_lb = np.uint8(utils.rle_decode(mask_lb, (h,w,1)))
            img_lb = img_lb.astype(np.float32) *  255.
            cv2.imwrite(f"{path}/large_bowel.png", img_lb)
            img_sb = np.uint8(utils.rle_decode(mask_sb, (h,w,1)))
            img_sb = img_sb.astype(np.float32) *  255.
            cv2.imwrite(f"{path}/small_bowel.png", img_sb)
            img_s = np.uint8(utils.rle_decode(mask_s, (h,w,1)))
            img_s = img_s.astype(np.float32) *  255.
            cv2.imwrite(f"{path}/stomach.png", img_s)
    
    def generate_images(self): 
        for _, row in self.df_clean.iterrows():
            case = str(row["case"])
            day = str(row["day"])
            slice = str(row['slice'])

            image_path_src = row["path"]
            image_path_dst = os.path.join(self.root, f"data/images/{case}/{day}/{slice}")
            if (not os.path.exists(image_path_dst)):
                os.makedirs(image_path_dst)

            shutil.copy(image_path_src, f"{image_path_dst}/img.png")       



def main():
    root = "uw_medison"
    data_cleaner = DataCleaner(root)
    data_cleaner.transform_dataframe()
    data_cleaner.generate_images()
    data_cleaner.generate_masks()
    



if __name__ == "__main__":
    main()
