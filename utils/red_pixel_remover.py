import os
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python red_pixel_remover.py <path_to_image>")
        exit(1)
    path = sys.argv[1]
    img = cv.imread(path, cv.IMREAD_UNCHANGED)  # BGR or BGRA
    if img is None:
        print("Could not open or find the image:", path)
        exit(1)
    if img.shape[2] != 4:
        print("The image must have an alpha channel.")
        exit(1)

    mask = cv.inRange(img[..., :3], (0, 0, 250, 0), (0, 0, 255, 255))
    img = cv.bitwise_and(img, img, mask=cv.bitwise_not(mask))

    import matplotlib.pyplot as plt

    plt.imshow(cv.cvtColor(img, cv.COLOR_BGRA2RGBA))
    plt.axis("off")
    plt.show()

    base_name, ext = os.path.splitext(os.path.basename(path))
    new_base_name = base_name + " (no red)"
    new_path = os.path.join(os.path.dirname(path), new_base_name + ext)
    cv.imwrite(new_path, img)
