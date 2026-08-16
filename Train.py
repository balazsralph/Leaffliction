# #####################################################################
# ######################       Train      #############################
# #####################################################################


import json
import argparse as args
import shutil
from Distribution import count_images, PICTURE_EXTENSIONS
from Augmentation import process_image
from pathlib import Path
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
import zipfile
import hashlib
from Distribution import display_error


def sha1_of(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def augment_train(train_dir):
    # 1. compter chaque classe
    counts = {d: count_images(d) for d in train_dir.iterdir() if d.is_dir()}
    # 2. cible globale = la plus grande classe
    maxi = max(counts.values())
    # 3. pour chaque classe sous la cible
    for class_dir, content in counts.items():
        if content >= maxi:
            continue
        needed = int((maxi - content) / 6)
        # 4. figer la liste des ORIGINAUX avant d'écrire
        originals = [f for f in class_dir.iterdir()
                     if f.is_file() and f.suffix.lower() in PICTURE_EXTENSIONS]
        # 5. augmenter les `needed` premiers, EN PLACE
        for f in originals[:needed]:
            process_image(f, True, class_dir)


def argparse():
    parser = args.ArgumentParser(prog="Train", description="Train a model")
    parser.add_argument("-src", required=True)
    parser.add_argument("-dst", required=True)
    return parser.parse_args()


def split_images(src, dst):
    for tree in src.iterdir():
        for type in tree.iterdir():
            train_dst = dst / "train" / type.name
            val_dst = dst / "val" / type.name
            Path.mkdir(train_dst, exist_ok=True,  parents=True)
            Path.mkdir(val_dst, exist_ok=True,  parents=True)
            k = int(count_images(type) * 0.2)
            files = [f for f in type.iterdir() if f.is_file()
                     and f.suffix.lower() in PICTURE_EXTENSIONS]
            for f in files[:k]:
                shutil.copy(f, val_dst / f.name)
            for f in files[k:]:
                shutil.copy(f, train_dst / f.name)


def prepare_data():
    train_ds = image_dataset_from_directory(
        "work_data/train",
        image_size=(128, 128),
        batch_size=32
    )
    val_ds = image_dataset_from_directory(
        "work_data/val",
        image_size=(128, 128),
        batch_size=32
    )
    class_names = train_ds.class_names
    return train_ds, val_ds, class_names


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
    parsed = argparse()
    src = Path(parsed.src)
    if not src.is_dir():
        display_error(f"Not a directory: {parsed.src}")
    dst = Path(parsed.dst)
    split_images(src, dst)
    train_path = dst / "train"
    augment_train(train_path)
    train_ds, val_ds, class_names = prepare_data()
    with open("class_names.json", "w") as f:
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
        filepath='model.keras',        # où sauvegarder
        monitor='val_loss',            # même métrique de référence
        save_best_only=True            # n'écrit que si ça s'améliore
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=[early_stop, checkpoint]
        )
    with zipfile.ZipFile("dataset.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write("model.keras")
        zf.write("class_names.json")
        for f in Path("work_data").rglob("*"):
            if f.is_file():
                zf.write(f)
    signature = sha1_of("dataset.zip")
    Path("signature.txt").write_text(signature + "\n")


if __name__ == "__main__":
    main()
