import sys
from pathlib import Path

import cv2 as cv
import numpy as np


def dhash(image: np.ndarray, hash_size: int = 8) -> int:
    """Compute the difference hash for an image.
    See https://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html).
    """
    resized = cv.resize(image, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum(2**i for (i, v) in enumerate(diff.flat) if v)


def find_duplicates(image_dir: str) -> None:
    """Find and print duplicates in the provided image directory."""
    hashes: dict[int, str] = {}

    for fn in Path(image_dir).rglob("*.*"):
        assert fn not in hashes, f"Duplicate filename `{fn}` found."

        if fn.suffix == ".txt":
            continue

        # open image
        img = cv.imread(fn, cv.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not open `{fn}` as image.")

        # compute hash
        hash = dhash(img)

        # check if hash is already in dictionary
        if hash in hashes:
            print(f"Possible duplicate: {hashes[hash]} <--> {fn}.")
        else:
            hashes[hash] = fn


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python duplicate_detector.py <path_to_dir>")
        exit(1)
    find_duplicates(sys.argv[1])
