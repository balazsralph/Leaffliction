# #####################################################################
# ######################       Train      #############################
# #####################################################################


import argparse as args
import sys
import shutil
from Distribution import count_images, PICTURE_EXTENSIONS
from Augmentation import process_batch
from pathlib import Path


def argparse():
    parser = args.ArgumentParser(prog="Train", description="Train a model")
    parser.add_argument("-src")
    parser.add_argument("-dst")
    return parser.parse_args()


def split_images(src, dst):
    for tree in src.iterdir():
        for type in tree.iterdir():
            train_dst = dst / "train" / tree.name / type.name
            val_dst = dst / "val" / tree.name / type.name
            Path.mkdir(train_dst, exist_ok=True,  parents=True)
            Path.mkdir(val_dst, exist_ok=True,  parents=True)
            k = int (count_images(type) * 0.2)
            files = [f for f in type.iterdir() if f.is_file() 
                     and f.suffix.lower() in PICTURE_EXTENSIONS]
            for f in files[:k]:
                shutil.copy(f, val_dst / f.name)
            for f in files[k:]:
                shutil.copy(f, train_dst / f.name)

def main():
    parsed = argparse()
    src = Path(parsed.src)
    dst = Path(parsed.dst)
    split_images(src, dst)
    train_path =  dst / "val"
    for tree in src.iterdir():
        process_batch(tree, train_path)

    print(src)


if __name__ == "__main__":
    main()
