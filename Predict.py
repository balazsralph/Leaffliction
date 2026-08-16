from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import argparse as args
from Distribution import PICTURE_EXTENSIONS, display_error
from Transformation import create_mask, apply_mask


def parse_args():
    parser = args.ArgumentParser(
        prog="Predict",
        description="use a trained model to make predictions"
        )
    parser.add_argument(
        'image',
        nargs="?",
        default=None)
    parser.add_argument('--model', default="model.keras")
    return parser.parse_args()


def display_result(img, label, confidence):
    mask = create_mask(img)
    masked = apply_mask(img, mask)
    images = [
        ('Original', img),
        ('Masked', masked)
    ]
    set_images(images)

    plt.suptitle(
        "\n====    DL Classification    ====",
        color="white",
        fontsize=24
    )
    plt.figtext(
        0.5,
        0.06,
        "Prediction : ",
        color="white",
        ha="right",
        fontsize=14
    )
    plt.figtext(
        0.5,
        0.06,
        f"{label}",
        color="lightgreen",
        ha="left",
        fontsize=14
    )
    plt.figtext(
        0.5,
        0.02,
        "Confidence : ",
        color="white",
        ha="right",
        fontsize=14
    )
    plt.figtext(
        0.5,
        0.02,
        f"{confidence:.2%}",
        color="lightgreen",
        ha="left",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()


def set_images(images: list[tuple[str, np.ndarray]]) -> None:
    n = len(images)
    fig, axes = plt.subplots(
        1, n, figsize=(4 * n, 6),
        facecolor='#242424',
        gridspec_kw={'wspace': 0.3},
    )
    if n == 1:
        axes = [axes]

    for ax, (title, img) in zip(axes, images):
        ax.imshow(img)
        ax.set_title(title, color="white")
        ax.axis("off")

    fig.subplots_adjust(top=0.85, bottom=0.15)
    return fig


def main():
    parsed = parse_args()
    model_path = Path(parsed.model)
    if not model_path.is_file():
        display_error("Model not found, set a path with --model option")

    try:
        model = load_model(model_path)
    except (ValueError, OSError) as e:
        display_error(f"Could not load model: {e}")

    with open("class_names.json") as f:
        class_names = json.load(f)
    if parsed.image is None:
        display_error("No image provided")
    path = Path(parsed.image)
    if path.suffix.lower() not in PICTURE_EXTENSIONS or not path.is_file():
        display_error("Not a picture")

    img = load_img(path, target_size=(128, 128))
    arr = img_to_array(img)  # (128, 128, 3), RGB
    batch = np.expand_dims(arr, axis=0)  # (1, 128, 128, 3)

    preds = model.predict(batch)  # (1, 7) : 7 probabilités
    idx = np.argmax(preds[0])  # indice de la plus forte
    label = class_names[idx]  # nom de la classe
    confidence = preds[0][idx]  # sa probabilité

    img_np = arr.astype(np.uint8)  # numpy uint8 pour plantcv/OpenCV
    display_result(img_np, label, confidence)


if __name__ == "__main__":
    main()
