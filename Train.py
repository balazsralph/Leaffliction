# #####################################################################
# ######################       Train      #############################
# #####################################################################
#
# Part 4 - Classification.
# Trains ONE CNN per plant directory, as the subject invokes it:
#     ./Train.py ./DATA/Apple/
# where the given directory holds the class subfolders directly
# (Apple/Apple_healthy/<images>, ...): a 2-level dataset.
#
# Packaging the final DB (zip + signature.txt over ALL trained plants)
# is a separate step, on purpose, so the signature covers every model:
#     make package


import json
import random
import shutil
import argparse
from pathlib import Path

from Distribution import count_images, display_error, PICTURE_EXTENSIONS
from Augmentation import process_image
from keras.utils import image_dataset_from_directory
from keras.models import Sequential
from keras.layers import (
    Rescaling,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    Input
)
from keras.callbacks import EarlyStopping, ModelCheckpoint


WORK_DIR = Path("work_directory")
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
VAL_RATIO = 0.2
SEED = 42
# process_image writes 6 augmentations per original (Rotate/Blur/Contrast/
# Scaling/Illumination/Projective), so one original ~= 6 new images.
AUG_PER_IMAGE = 6


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Train",
        description="Train a CNN on one plant directory "
                    "(e.g. ./Train.py ./DATA/Apple/)",
    )
    parser.add_argument(
        "src",
        help="plant directory containing class subfolders",
    )
    return parser.parse_args()


def list_images(directory):
    return [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in PICTURE_EXTENSIONS
    ]


def split_images(src, train_root, val_root):
    """2-level dataset src/<class>/<images>: split each class into train/val.

    The split is shuffled with a fixed seed (reproducible, unbiased) and the
    val ratio is applied to the actual image list, not a separate count.
    """
    rng = random.Random(SEED)
    for class_dir in sorted(src.iterdir()):
        if not class_dir.is_dir():
            continue
        files = list_images(class_dir)
        if not files:
            continue
        rng.shuffle(files)
        k = int(len(files) * VAL_RATIO)
        train_dst = train_root / class_dir.name
        val_dst = val_root / class_dir.name
        train_dst.mkdir(parents=True, exist_ok=True)
        val_dst.mkdir(parents=True, exist_ok=True)
        for f in files[:k]:
            shutil.copy(f, val_dst / f.name)
        for f in files[k:]:
            shutil.copy(f, train_dst / f.name)


def augment_train(train_root):
    """Balance the TRAIN classes up to the largest one (val untouched)."""
    counts = {d: count_images(d) for d in train_root.iterdir() if d.is_dir()}
    if not counts:
        display_error(f"No class found to augment in: {train_root}")
    maxi = max(counts.values())
    for class_dir, content in counts.items():
        if content >= maxi:
            continue
        needed = (maxi - content) // AUG_PER_IMAGE
        originals = list_images(class_dir)
        for f in originals[:needed]:
            process_image(f, True, class_dir)


def prepare_data(train_root, val_root):
    train_ds = image_dataset_from_directory(
        train_root,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = image_dataset_from_directory(
        val_root,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    return train_ds, val_ds, train_ds.class_names


def build_model(class_names):
    model = Sequential([
        Input(shape=(128, 128, 3)),
        Rescaling(1./255),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(class_names), activation='softmax')
    ])
    model.summary()
    return model


def main():
    parsed = parse_args()
    src = Path(parsed.src)
    if not src.is_dir():
        display_error(f"Not a directory: {parsed.src}")

    plant = src.name
    plant_work = WORK_DIR / plant
    train_root = plant_work / "train"
    val_root = plant_work / "val"

    # Fresh split for this plant so re-running doesn't stack copies/augments.
    if plant_work.exists():
        shutil.rmtree(plant_work)

    split_images(src, train_root, val_root)
    augment_train(train_root)

    train_ds, val_ds, class_names = prepare_data(train_root, val_root)

    model_path = Path(f"model_{plant}.keras")
    classes_path = Path(f"class_names_{plant}.json")
    with open(classes_path, "w") as f:
        json.dump(class_names, f, indent=2)

    model = build_model(class_names)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    checkpoint = ModelCheckpoint(
        filepath=str(model_path),      # où sauvegarder
        monitor='val_loss',            # même métrique de référence
        save_best_only=True            # n'écrit que si ça s'améliore
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=[early_stop, checkpoint]
    )

    print(f"Model saved   : {model_path}")
    print(f"Classes saved : {classes_path}")
    print("When every plant is trained, build the DB: make package")


if __name__ == "__main__":
    main()
