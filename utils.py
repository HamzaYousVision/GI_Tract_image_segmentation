import numpy as np
import yaml
from types import SimpleNamespace
import json
def rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape, color=1):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros((shape[0] * shape[1], shape[2]), dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = color
    return img.reshape(shape)

def parse_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        my_namespace = json.loads(json.dumps(
            config_dict), object_hook=lambda item: SimpleNamespace(**item))

        return my_namespace

def parse_yaml_config(config_path):
    config = parse_yaml(config_path)
    return config

def visualize_samples():
    # TODO: visualize samples from dataset. Will be used for validating the Dataset and also to visualize the results of the model during training