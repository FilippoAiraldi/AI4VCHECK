import argparse
from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np


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
    assert len(cnts) == 1, "Expected only one contour."
    cnt = cnts[0]
    M = cv.moments(cnt)
    area = M["m00"]  # r = np.sqrt(area / np.pi) is not robust to outliers
    center = np.asarray((M["m10"], M["m01"]), dtype=float) / area
    r = np.mean([np.linalg.norm(p - center) for p in cnt.squeeze(1)])
    return center, r


if __name__ == "__main__":
    max_circles = 5
    parser = argparse.ArgumentParser(
        description="Extract circles from corneal image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("img", type=str, help="Filepath of image")
    parser.add_argument(
        "circle", type=int, choices=range(1, max_circles + 1), help="Circle to extract"
    )
    args = parser.parse_args()

    # read image
    img_path = Path(args.img)
    img = cv.imread(img_path, cv.IMREAD_UNCHANGED)  # BGR or BGRA
    if img is None:
        print("Could not open or find the image:", img_path)
        exit(1)

    # calculate the enclosing circle
    center, r = calculate_enclosing_circle(img)
    center = center.astype(int)

    # extract the image from the specified enclosing circle
    n = args.circle
    mask = np.empty(img.shape[:2], np.uint8)
    cv.circle(mask, center, int(r / max_circles * n), 255, cv.FILLED)
    subimg = cv.bitwise_and(img, img, mask=mask)

    # save subimage to disk
    cv.imwrite(img_path.with_stem(f"{img_path.stem} (circle {n})"), subimg)
