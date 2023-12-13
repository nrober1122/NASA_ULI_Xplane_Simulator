import os
import tqdm
import pathlib

import h5py
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from matplotlib.colors import CenteredNorm
from PIL import Image
from torch.utils.data import DataLoader

from simulation.controllers import get_p_control_torch, getProportionalControl
from simulation.tiny_taxinet import downsample_image
from simulation.tiny_taxinet2 import get_network
from tiny_taxinet_train.model_tiny_taxinet import TinyTaxiNetDNN
from tiny_taxinet_train.tiny_taxinet_dataloader import tiny_taxinet_prepare_dataloader

NASA_ULI_ROOT_DIR = os.environ["NASA_ULI_ROOT_DIR"]
data_dir = pathlib.Path(os.environ["NASA_ULI_DATA_DIR"])
data_folder = data_dir / "morning"


def main():
    strides = [1, 2, 4, 8]

    for folder in data_folder.iterdir():
        stride_images = {k: [] for k in strides}
        stride_labels = {k: [] for k in strides}

        # Read label in folder.
        label_csv = folder / "labels.csv"
        labels = np.genfromtxt(label_csv, delimiter=",", skip_header=1, dtype=None, encoding=None)
        for raw_label in tqdm.tqdm(labels):
            raw_label = list(raw_label)
            image_name = raw_label[0]
            # Extract the label.
            # 0: absolute_time_GMT_seconds
            # 1: relative_time_seconds
            # 2: distance_to_centerline_meters
            # 3: distance_to_centerline_NORMALIZED
            # 4: downtrack_position_meters
            # 5: downtrack_position_NORMALIZED
            # 6: heading_error_degrees
            # 7: heading_error_NORMALIZED
            # 8: period_of_day
            # 9: cloud_type
            label = np.array(raw_label[1:])
            # We want
            # [ distance_to_centerline_meters; heading_error_degrees; downtrack_position_meters ]
            label = label[[2, 6, 4]]

            # Read image.
            image_path = folder / image_name
            assert image_path.exists()

            # Image has already been resized, but has not been converted to grayscale, nor has been cropped.
            image = Image.open(image_path).convert("L")
            assert image.width == 360 and image.height == 200

            # Crop out nose, sky, bottom of image.
            image = image.crop((55, 5, 360, 135)).resize((256, 128))
            image = np.array(image) / 255.0
            assert image.shape == (128, 256)

            for stride in strides:
                img_downsampled = downsample_image(image, stride)
                stride_images[stride].append(img_downsampled)
                stride_labels[stride].append(label)

        # Save the entire dataset as h5.
        for stride in strides:
            stride_images[stride] = np.stack(stride_images[stride], axis=0)
            stride_labels[stride] = np.stack(stride_labels[stride], axis=0)

            # Save the dataset.
            h5_file = data_dir / f"morning_downsampled_stride{stride}/{folder.name}_stride{stride}.h5"
            h5_file.parent.mkdir(exist_ok=True, parents=True)
            with h5py.File(h5_file, "w") as f:
                f.create_dataset("X_train", data=stride_images[stride])
                f.create_dataset("y_train", data=stride_labels[stride])
            logger.info("Saved to {}", h5_file)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
