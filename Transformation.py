import io
import sys
import cv2
import numpy as np
from PIL import Image
import argparse as args
from pathlib import Path
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
from Distribution import PICTURE_EXTENSIONS


def show_images(images: list[tuple[str, np.ndarray]], hist_img) -> None:
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 6)          # grille 2 lignes × 6 colonnes

    # ligne 0 : les 6 images, une par colonne
    for i, (title, im) in enumerate(images):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(im, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    # ligne 1 : l'histo, étalé sur TOUTES les colonnes
    axh = fig.add_subplot(gs[1, :])      # ':' = toutes les colonnes
    axh.imshow(hist_img)
    axh.set_title("Color histogram")
    axh.axis('off')

    plt.tight_layout()
    plt.show()


def gaussian_blur(mask):
    return pcv.gaussian_blur(img=mask, ksize=(5, 5), sigma_x=0, sigma_y=None)


def apply_mask(img, mask):
    masked = pcv.apply_mask(img=img, mask=mask, mask_color='white')
    return masked


def create_mask(img):
    gray = pcv.rgb2gray_lab(rgb_img=img, channel='a')
    mask = pcv.threshold.otsu(gray_img=gray, object_type='dark')
    mask = pcv.fill(bin_img=mask, size=200)
    mask = pcv.fill_holes(bin_img=mask)
    return mask


def region_of_interest(img, mask):
    h, w = img.shape[:2]
    roi = pcv.roi.rectangle(img=img, x=0, y=0, h=h, w=w)
    filtered = pcv.roi.filter(mask=mask, roi=roi, roi_type='largest')
    # 1. extraire le contour de la feuille depuis le masque filtré
    contours, _ = cv2.findContours(
        filtered, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    # 2. copie de l'image pour dessiner dessus
    overlay = img.copy()
    # 3. remplir la feuille en vert (thickness = -1 = rempli)
    filtered = cv2.drawContours(overlay, contours, -1, (0, 255, 0), -1)
    # 4. rectangle bleu = la ROI
    filtered = cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), (0, 0, 255), 5)
    return filtered


def analysis(img, mask):
    pcv.params.text_size = 0.0001
    pcv.params.text_thickness = 0
    analysis_image = pcv.analyze.size(img=img, labeled_mask=mask, n_labels=1)
    return analysis_image


def pseudolandmarks(img, mask):
    top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
        img=img,
        mask=mask
    )
    overlay = img.copy()
    for p in top:
        x, y = int(p[0][0]), int(p[0][1])
        cv2.circle(overlay, (x, y), 3, (255, 0, 0), -1)  # bleu (BGR)
    for p in bottom:
        x, y = int(p[0][0]), int(p[0][1])
        cv2.circle(overlay, (x, y), 3, (255, 0, 255), -1)  # magenta
    for p in center_v:
        x, y = int(p[0][0]), int(p[0][1])
        cv2.circle(overlay, (x, y), 3, (0, 165, 255), -1)  # orange
    return overlay


def histogram(img, mask):
    fig = plt.figure(figsize=(12, 6))
    bgr = img  # déjà en BGR : donne blue, green, red
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)   # donne hue, saturation, value
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)   # donne lightness, a, b
    b, g, r = cv2.split(bgr)         # blue, green, red
    h, s, v = cv2.split(hsv)         # hue, saturation, value
    l, a, bb = cv2.split(lab)        # lightness, green-magenta, blue-yellow
    channels = [
        (b,  "blue",          "blue"),
        (g,  "green",         "green"),
        (r,  "red",           "red"),
        (h,  "hue",           "purple"),
        (s,  "saturation",    "cyan"),
        (v,  "value",         "orange"),
        (l,  "lightness",     "gray"),
        (a,  "green-magenta", "magenta"),
        (bb, "blue-yellow",   "gold"),
    ]

    total = cv2.countNonZero(mask)      # nombre de pixels de la feuille

    for channel_img, name, color in channels:
        # compte les pixels par intensité, UNIQUEMENT dans la feuille
        hist = cv2.calcHist([channel_img], [0], mask, [256], [0, 256])
        # convertit les comptes en pourcentage du total
        hist = (hist / total) * 100
        # trace la courbe
        plt.plot(hist, color=color, label=name)

    plt.xlabel("Pixel intensity")
    plt.ylabel("Proportion of pixels (%)")
    plt.legend(title="color Channel")
    plt.title("Color histogram")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf))


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
        # print(save_path)
        Image.fromarray(img).save(save_path)
        # print(f"Saved {save_path}")
        

def process_image(file, save, dst=None, only_mask=False):
    original = np.array(Image.open(file))
    img, path, name = pcv.readimage(file)
    mask = create_mask(img)
    hist = histogram(img, mask)
    masked = apply_mask(img, mask)

    images = [
        ("Original", original),
        ('Gaussian blur', gaussian_blur(mask)),
        ("masked", masked),
        ("ROI Objects", region_of_interest(img, mask)),
        ("Analyze Object", analysis(img, mask)),
        ("Pseudolandmarks", pseudolandmarks(img, mask))
    ]

    if only_mask:
        cv2.imwrite(f"{dst}/{file.stem}_Mask.JPG", masked)
    elif save:
        save_images(images, file, dst)
    else:
        show_images(images, hist)


def parseargs():
    parser = args.ArgumentParser(
        prog="Transformation",
        description="this programme transform pictures to analyse data"
        )
    parser.add_argument(
        'image',
        nargs='?',
        default=None,
        help='path to a single image (display mode)'
        )
    parser.add_argument('-src', help='source directory (batch mode)')
    parser.add_argument('-dst', help='destination directory (batch mode)')
    parser.add_argument(
        '-mask',
        action='store_true',
        help='save only the mask'
        )
    return parser.parse_args()


def main() -> None:
    parsed = parseargs()

    if parsed.image:
        picture_path = Path(parsed.image)
        if picture_path.suffix.lower() not in PICTURE_EXTENSIONS \
                or not picture_path.is_file():
            print("Not a picture")
            sys.exit(1)
        process_image(picture_path, save=False)

    elif parsed.src and parsed.dst:
        src = Path(parsed.src)
        dst = Path(parsed.dst)
        dst.mkdir(parents=True, exist_ok=True)
        for file in src.iterdir():
            if file.suffix.lower() in PICTURE_EXTENSIONS:
                process_image(file, save=True, dst=dst, only_mask=parsed.mask)
    else:
        print("Wrong arguments")
        sys.exit(1)


if __name__ == "__main__":
    main()
