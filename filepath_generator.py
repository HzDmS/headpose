# filename_list.txt genrator for 300W_LP dataset.

import os
import glob
import argparse

SUBFOLDERS = ['AFW', 'HELEN', 'LFPW', 'IBUG']

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_path",
    type=str,
    help="path of 300W_LP dataset."
)

args = parser.parse_args()

with open(os.path.join(args.dataset_path, "filename_list.txt"), "w") as writer:
    for folder in SUBFOLDERS:
        files = glob.glob(os.path.join(args.dataset_path, folder, "*.jpg"))
        for f in files:
            base_name = os.path.basename(f).split(".")[0]
            writer.write(
                os.path.join(args.dataset_path, folder, base_name) + "\n")
            writer.write(
                os.path.join(args.dataset_path, folder + "_Flip", base_name) + "\n")
