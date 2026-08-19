import sys
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from Distribution import (
    count_images,
    collect_by_plant,
    display_error,
    PICTURE_EXTENSIONS
)


# ###########################################################
# #########             AUGMENTATION                #########
# ###########################################################


def rotate_image(image: np.ndarray, save) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    scale = 0.75
    if save:
        scale = 1
    matrix = cv2.getRotationMatrix2D(center, -25, scale)
    return cv2.warpAffine(
        image, matrix, (w, h),
        # borderMode=cv2.BORDER_REFLECT,
        borderValue=(255, 255, 255),
    )


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

    src = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ])
    dst = np.float32([
        [w * 0.15, h * 0.1],
        [w * 0.85, h * 0.05],
        [w * 0.95, h * 0.95],
        [w * 0.05, h * 0.9],
    ])

    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        image, matrix, (w, h), borderValue=(255, 255, 255),
    )


# ###########################################################
# #########              FUNCTIONS                  #########
# ###########################################################

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


def copy_images(dir, new_path):
    for file in dir.iterdir():
        shutil.copy(file, new_path)


def save_images(
        images: list[tuple[str, np.ndarray]],
        picture_path: Path,
        new_path
        ) -> None:
    for title, img in images:
        if title == "Original":
            continue

        filename = f"{picture_path.stem}_{title}{picture_path.suffix}"
        save_path = new_path / filename
        Image.fromarray(img).save(save_path)


def process_image(file, save, new_path):

    original = np.array(Image.open(file))
    images = [
        ("Original", original),
        ("Rotate", rotate_image(original, save)),
        ("Blur", blur_image(original)),
        ("Contrast", contrast_image(original)),
        ("Scaling", scaling_image(original)),
        ("Illumination", illumination_image(original)),
        ("Projective", projective_image(original)),
    ]

    if save:
        save_images(images, file, new_path)
    else:
        show_images(images)


def process_batch(picture_path, dst, tree=True):
    plants = collect_by_plant(picture_path)
    if picture_path.stem not in plants:
        display_error(
            f"No plant class found in: {picture_path} "
            f"(expected subfolders like 'Apple_healthy')"
        )
    maxi = int(max(plants[picture_path.stem].values()))

    for dir in picture_path.iterdir():
        if tree:
            new_dir = Path(dst) / picture_path.stem / dir.stem
        else:
            new_dir = Path(dst) / dir.stem
        new_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        content = count_images(dir)
        max_to_augment = int((maxi - content) / 6)
        copy_images(dir, new_dir)
        print("images in", dir.stem, content)
        if content == maxi:
            print('skipping dir')
            continue
        print("max images : ", maxi)
        print("missing : ", max_to_augment)
        for file in dir.iterdir():
            if (file.suffix.lower() not in PICTURE_EXTENSIONS
                    or not file.is_file()):
                continue
            process_image(file, True, new_dir)
            count = count + 1
            if count == max_to_augment:
                break


# ###########################################################
# #########                 MAIN                    #########
# ###########################################################


def main() -> None:
    if len(sys.argv) != 2:
        print("Wrong number of arguments")
        sys.exit(1)

    picture_path = Path(sys.argv[1])

    if picture_path.is_file():
        process_image(picture_path, False, None)
        sys.exit(0)
    else:
        process_batch(picture_path, "augmented_directory")


if __name__ == "__main__":
    main()
