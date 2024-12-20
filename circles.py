import argparse
from pathlib import Path
from typing import Optional
from warnings import warn

import cv2 as cv
import numpy as np

N_CIRCLES = 5


def calculate_enclosing_circle(
    img: np.ndarray, gray_img: Optional[np.ndarray] = None
) -> tuple[np.ndarray, float]:
    """Computes the circle that encloses the corneal segmented image.

    Parameters
    ----------
    img : np.ndarray
        A 3- or 4-channel image containing the cornea. If the image has 4 channels,
        areas outside of the cornea should be transparent. If the image has 3 channels,
        areas outside of the cornea should be black.
    gray_img : np.ndarray, optional
        The grayscale version of the image. If not provided, it is computed by
        converting the image to grayscale.

    Returns
    -------
    center and radiius
        The center and radius of the circle that encloses the corneal segmented image.
    """
    if img.shape[2] == 4:
        _, thresholded_img = cv.threshold(img[..., 3], 0, 255, cv.THRESH_BINARY)
    else:
        if gray_img is None:
            gray_img = cv.cvtColor(cv.COLOR_BGR2GRAY)
        blurred_image = cv.GaussianBlur(gray_img, (71, 71), 11)
        _, thresholded_img = cv.threshold(
            blurred_image, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU
        )

    cnts, _ = cv.findContours(thresholded_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 1:
        cnt = cnts[0]
    else:
        warn(f"Expected only one contour, but found {len(cnts)}.")
        cnt = max(cnts, key=cv.contourArea)
        # import matplotlib.pyplot as plt
        # cv.drawContours(img, [cnt], 0, [0, 255, 0, 255], 20)
        # plt.imshow(img)
        # plt.axis("off")
        # plt.show()

    M = cv.moments(cnt)
    area = M["m00"]  # r = np.sqrt(area / np.pi) is not robust to outliers
    center = np.asarray((M["m10"], M["m01"]), dtype=float) / area
    r = np.linalg.norm(cnt.squeeze(1) - center, axis=1).mean()
    return center, r


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract circles from corneal image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("img", type=Path, help="Filepath of image")
    parser.add_argument(
        "circles",
        choices=list(map(str, range(1, N_CIRCLES + 1))),
        type=str,
        nargs="+",
        help="Circles to extract",
    )
    args = parser.parse_args()

    # read image
    path = args.img
    img = cv.imread(path, cv.IMREAD_UNCHANGED)  # BGR or BGRA
    if img is None:
        print("Could not open or find the image:", path)
        exit(1)

    # calculate the enclosing circle
    center, r = calculate_enclosing_circle(img)
    center = center.astype(int)

    # extract the image from the specified enclosing circles
    subimgs = []
    for n in map(int, args.circles):
        mask = np.empty(img.shape[:2], np.uint8)
        cv.circle(mask, center, int(r / N_CIRCLES * n), 255, cv.FILLED)
        subimg = cv.bitwise_and(img, img, mask=mask)
        subimgs.append(subimg)

    # save subimages to disk
    circle_folder = path.parent / "circles"
    circle_folder.mkdir(exist_ok=True)
    for n, subimg in zip(args.circles, subimgs):
        new_path = circle_folder / f"{path.stem}-circle-{n}{path.suffix}"
        cv.imwrite(new_path, subimg)
