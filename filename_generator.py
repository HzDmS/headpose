# filename_list.txt genrator for 300W_LP dataset.

import os
import glob
import argparse
import io
import numpy as np

from utils import get_ypr_from_mat

SUBFOLDERS = ['AFW', 'HELEN', 'LFPW', 'IBUG']

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    help="name of the dataset"
)
parser.add_argument(
    "--dataset_path",
    type=str,
    help="path of dataset."
)

args = parser.parse_args()


def is_filtered(mat_path):

    """ filter the data if any pose angle is greater than 99 degree \
        or smaller than -99 degree.

    Parameters
    ----------
    mat_path: str, path of the .mat file.
    """

    pose = get_ypr_from_mat(mat_path)
    pitch = pose[0] * 180 / np.pi
    yaw = pose[1] * 180 / np.pi
    roll = pose[2] * 180 / np.pi

    ret = (np.abs(pitch) > 99 or
           np.abs(yaw) > 99 or
           np.abs(roll) > 99)

    return ret


def list_images(path, writer, folder=None, clean=False):

    """ List all files in the dataset, and write them into a text file.

    Parameters
    ----------
    path: str, path to the dataset.
    writer: text writer.
    folder: str or None, str if images are stored in subfolders, else None.
    filter: bool, True if the dataset needs to be filtered, else False.

    """

    if not isinstance(writer, io.TextIOWrapper):
        raise TypeError("writer is not a instance of io.TextIOWrapper")

    files = glob.glob(os.path.join(path, "*.jpg"))
    for f in files:
        base_name = os.path.basename(f).split(".")[0]
        if clean and is_filtered(os.path.join(path, base_name + '.mat')):
            continue
        if not folder:
            writer.write(base_name + "\n")
        else:
            writer.write(
                os.path.join(folder, base_name) + "\n")


with open(os.path.join(args.dataset_path, "filename_list.txt"), "w") as writer:

    if args.dataset == "300W_LP":
        for folder in SUBFOLDERS:
            list_images(
                os.path.join(args.dataset_path, folder),
                writer,
                folder=folder,
            )

    elif args.dataset == "AFLW2000":
        list_images(os.path.join(args.dataset_path), writer, clean=True)

    else:
        AttributeError("{} is not valid".format(args.dataset))
