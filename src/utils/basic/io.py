import logging
import os
from typing import List

import numpy as np
from skimage.io import imread, imsave

from src.utils.basic.base import sorted_nicely


def get_file_list(
    root_dir: str,
    absolute_path: bool = True,
    file_ending: bool = True,
    file_type_filter: str = None,
) -> List:
    if not os.path.exists(root_dir):
        logging.debug("Path not found.")
        return []
    list_of_data_locs = []
    for (root_dir, dirname, filename) in os.walk(root_dir):
        for file in filename:
            if file_type_filter is not None and file_type_filter not in file:
                continue
            else:
                if not file_ending:
                    file = file[: file.index(".")]
                if absolute_path:
                    list_of_data_locs.append(
                        os.path.abspath(os.path.join(root_dir, file))
                    )
                else:
                    list_of_data_locs.append(file)
    return sorted_nicely(list_of_data_locs)


def read_tiff_from_disk(file: str) -> np.ndarray:
    img = np.array(imread(file))
    return img


def save_img_as_tiff(img: np.ndarray, file: str):
    imsave(file, arr=img)


def save_figure_as_png(fig, file):
    fig.savefig(file)
