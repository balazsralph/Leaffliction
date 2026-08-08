#!/usr/bin/env python3
"""Part 1 — Analyse the dataset and plot pie/bar charts per plant type."""

import sys
from pathlib import Path
import matplotlib.pyplot as plt

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


###########################################################
#########          FUNCTIONS DEFINITIONS         #########
###########################################################

def count_images(directory: Path) -> int:
    total = 0
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            total += 1
    return total


def collect_by_plant(root: Path) -> dict[str, dict[str, int]]:
    plants = {}

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue

        n_images = count_images(subdir)
        if n_images == 0:
            continue

        # "Apple_healthy" → plant = "Apple"
        file_name = subdir.name.split("_", 1)
        plant_name = file_name[0]
        class_name = file_name[1]

        if plant_name not in plants:
            plants[plant_name] = {}

        plants[plant_name][class_name] = n_images

    return plants


def plot_plant(plant: str, classes: dict[str, int]) -> None:
    labels = list(classes.keys())
    values = list(classes.values())
    colors = plt.cm.tab10.colors[: len(labels)]

    fig, (pie, bar) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Distribution — {plant}")

    pie.pie(values, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    pie.set_title("Pie chart")

    bar.bar(labels, values, color=colors)
    bar.set_title("Bar chart")
    bar.set_ylabel("Number of images")
    bar.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.show()



###########################################################
#########                 MAIN                   #########
###########################################################

def main() -> None:
    if len(sys.argv) != 2:
        print("Wrong number of arguments")
        sys.exit(1)

    root = Path(sys.argv[1])
    if not root.is_dir():
        print("Not a directory")
        sys.exit(1)

    plants = collect_by_plant(root)
    if not plants:
        print("No image classes found")
        sys.exit(1)

    for plant, classes in plants.items():
        total = sum(classes.values())
        print(f"\n{plant} ({total} images)")
        for name, count in classes.items():
            print(f"  {name}: {count}")
        plot_plant(plant, classes)


if __name__ == "__main__":
    main()
