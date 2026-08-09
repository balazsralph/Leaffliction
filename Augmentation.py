import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

PICTURE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
	".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF"
}

###########################################################
#########             AUGMENTATION                #########
###########################################################

def rotate_image(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, -45, 1)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT, borderValue=(255, 255, 255))

def blur_image(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (15, 15), 0)

def contrast_image(image: np.ndarray) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=1.5)

def scaling_image(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    scale = 1.5
    enlarged = cv2.resize(image, (int(w * scale), int(h * scale)))

    new_h, new_w = enlarged.shape[:2]
    y = (new_h - h) // 2
    x = (new_w - w) // 2
    return enlarged[y:y + h, x:x + w]

def illumination_image(image: np.ndarray) -> np.ndarray:
    return cv2.convertScaleAbs(image, beta=60)

def projective_image(image: np.ndarray) -> np.ndarray:
    """Déforme l'image avec une perspective (effet 'vu de biais')."""
    h, w = image.shape[:2]

    # 4 coins de l'image d'origine
    src = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ])
    # 4 coins déplacés
    dst = np.float32([
        [w * 0.15, h * 0.1],
        [w * 0.85, h * 0.05],
        [w * 0.95, h * 0.95],
        [w * 0.05, h * 0.9],
    ])

    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (w, h), borderValue=(255, 255, 255))

###########################################################
#########              FUNCTIONS                  #########
###########################################################

def show_images(images: list[tuple[str, np.ndarray]]) -> None:
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (title, img) in zip(axes, images):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

###########################################################
#########                 MAIN                   #########
###########################################################

def main() -> None:
    if len(sys.argv) != 2:
        print("Wrong number of arguments")
        sys.exit(1)

    picture_path = Path(sys.argv[1])
    if picture_path.suffix.lower() not in PICTURE_EXTENSIONS or not picture_path.is_file():
        print("Not a picture")
        sys.exit(1)

    original = np.array(Image.open(picture_path))

    images = [
        ("Original", original),
        ("Rotate", rotate_image(original)),
        ("Blur", blur_image(original)),
        ("Contrast", contrast_image(original)),
        ("Scaling", scaling_image(original)),
        ("Illumination", illumination_image(original)),
        ("Projective", projective_image(original)),
    ]
    show_images(images)


if __name__ == "__main__":
    main()
