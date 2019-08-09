# filename_list.txt genrator for 300W_LP dataset.

import os
import glob
import argparse
import io

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


def list_images(path, writer, folder=None):

    """ List all files in the dataset, and write them into a text file.

    Parameters
    ----------
    path: str, path to the dataset.
    writer: text writer.
    folder: str or None, str if images are stored in subfolders, else None.

    """

    if not isinstance(writer, io.TextIOWrapper):
        raise TypeError("writer is not a instance of io.TextIOWrapper")

    files = glob.glob(os.path.join(path, "*.jpg"))
    for f in files:
        base_name = os.path.basename(f).split(".")[0]
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
        list_images(os.path.join(args.dataset_path), writer)

    else:
        AttributeError("{} is not valid".format(args.dataset))
