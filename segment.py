import argparse
import json
from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parses the command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="VCHECK segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("img", type=str, help="Filepath of image to segment")
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prior to saving to disk",
    )
    group = parser.add_argument_group("Pre-processing options")
    group.add_argument(
        "--rotate",
        type=float,
        default=0.0,
        help="Rotation of the original image in degrees",
    )
    group = parser.add_argument_group("Segmentation options")
    group.add_argument("--clahe", type=bool, default=False)
    group.add_argument(
        "--adaptive-thres-size",
        type=int,
        default=101,
        help="Adaptive thresholding neighborhood size",
    )
    group.add_argument(
        "--adaptive-thres-const",
        type=float,
        default=3.0,
        help="Adaptive thresholding constant",
    )
    group.add_argument(
        "--args-json",
        type=str,
        default="",
        help="Json with arguments to be loaded",
    )
    args = parser.parse_args()

    # load arguments from json - if not provided, see if there is a file named args.json
    args.img = Path(args.img)
    if not args.args_json:
        args_json = args.img.with_name("args.json")
        if args_json.is_file():
            args.args_json = args_json
    if args.args_json:
        with open(args.args_json, "r") as file:
            data = json.load(file)
            for key, value in data.items():
                setattr(args, key, value)
    return args


def preprocess(img: np.ndarray, rotate: float) -> np.ndarray:
    """Pre-processes the image prior to segmentation.

    Parameters
    ----------
    img : np.ndarray
        A 3- or 4-channel image containing the cornea. If the image has 4 channels,
        areas outside of the cornea should be transparent. If the image has 3 channels,
        areas outside of the cornea should be black.
    rotate : float
        Rotation of the original image in degrees.

    Returns
    -------
    np.ndarray
        The pre-processed image.
    """
    if rotate != 0.0:
        rows, cols = img.shape[:2]
        M = cv.getRotationMatrix2D((cols / 2, rows / 2), rotate, 1.0)
        img = cv.warpAffine(
            img, M, (cols, rows), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_TRANSPARENT
        )
    return img


def find_contours_TB_pixels(
    img: np.ndarray,
    gray_img: Optional[np.ndarray] = None,
    clahe: bool = False,
    adaptive_thres_size: int = 101,
    adaptive_thres_const: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    """Finds the contours of the Trypan Blue-stained (TB) regions in the corneal image
    (with the cornea already segmented out) via the Watershed algorithm.

    Parameters
    ----------
    img : np.ndarray
        A 3- or 4-channel image containing the cornea. If the image has 4 channels,
        areas outside of the cornea should be transparent. If the image has 3 channels,
        areas outside of the cornea should be black.
    gray_img : np.ndarray, optional
        The grayscale version of the image. If not provided, it is computed by
        converting the image to grayscale.
    clahe : bool, optional
        Whether to perform Contrast Limited Adaptive Histogram Equalization (CLAHE) on
        the image before segmenting it.
    adaptive_thres_size : int, optional
        Size of the adaptive thresholding neighborhood. It must be an odd integer.
    adaptive_thres_const : float, optional
        Constant subtracted from the mean or weighted mean to compute the threshold
        value.

    Returns
    -------
    corneal mask, and TB-positive mask and contours
        The mask of the corneal pixeles (as an array), the TB-positive mask (as an
        array) of the regions that are believed to be stained by the TB dye and the
        watershed-segmented contours of these regions (as a tuple of arrays).
    """
    # first of all, convert to grayscale and compute the mask of the cornea (i.e.,
    # separate pixels within the cornea from pixels outside of it)
    has_four_channels = img.shape[2] == 4
    if gray_img is None:
        gray_img = cv.cvtColor(
            img, cv.COLOR_BGRA2GRAY if has_four_channels else cv.COLOR_BGR2GRAY
        )
    _, corneal_mask = cv.threshold(
        img[..., 3] if has_four_channels else gray_img, 0, 255, cv.THRESH_BINARY
    )

    # convert to grayscale, remove noise and smooth image, and apply a first threshold
    # to coarsely extract all suspected TB-positive pixels
    if clahe:  # NOTE: tunable (!!!) - bool: perform CLAHE or not
        size = min(img.shape[:2]) // 100
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(size, size))
        gray_img = clahe.apply(gray_img)
    blurred_img = cv.GaussianBlur(
        gray_img,
        (9, 9),  # NOTE: tunable (!) - pairs of positive integers
        0,  # NOTE: tunable (!) - nonnegative float
    )
    thresholded_img = cv.adaptiveThreshold(
        blurred_img,
        255,
        cv.ADAPTIVE_THRESH_MEAN_C,
        cv.THRESH_BINARY_INV,
        adaptive_thres_size,  # NOTE: tunable (!!) - odd integer
        adaptive_thres_const,  # NOTE: tunable (!!!) - float
    )
    # _, axs = plt.subplots(1, 2, constrained_layout=True, sharex=True, sharey=True)
    # axs[0].imshow(blurred_img, cmap="gray")
    # axs[1].imshow(thresholded_img, cmap="gray")
    # for ax in axs.flat:
    #     ax.set_axis_off()
    # plt.show()

    # segment the image via the Watershed algorithm, according to the tutorial found at
    # https://docs.opencv.org/3.4/d3/db4/tutorial_py_watershed.html.

    # first, perform an opening to get rid of small thresholded nonzero regions
    kernel = cv.getStructuringElement(
        cv.MORPH_RECT,
        (3, 3),  # NOTE: tunable (!!) - pairs of positive integers (odd?)
    )
    mask = cv.morphologyEx(
        thresholded_img,
        cv.MORPH_OPEN,
        kernel,
        iterations=1,  # NOTE: tunable (!) - positive integer
    )
    # plt.imshow(mask, cmap="gray")
    # plt.axis("off")
    # plt.show()

    # then, for each nonzero pixel compute the distance to the nearest zero pixel, and
    # use it to discern foreground, background and unknown pixels
    distance = cv.distanceTransform(mask, cv.DIST_L2, cv.DIST_MASK_3)
    _, foreground = cv.threshold(
        distance,
        0.1 * distance.max(),  # NOTE: tunable (!!!) - threshold multiplier
        255,
        cv.THRESH_BINARY,
    )
    foreground = foreground.astype(np.uint8)
    background = cv.dilate(
        mask,
        kernel,
        iterations=3,  # NOTE: tunable (?) - nonnegative integer
    )
    unknown = cv.subtract(background, foreground)
    # _, axs = plt.subplots(2, 2, constrained_layout=True, sharex=True, sharey=True)
    # axs[0, 0].imshow(distance, cmap="gray")
    # axs[0, 1].imshow(foreground, cmap="gray")
    # axs[1, 0].imshow(background, cmap="gray")
    # axs[1, 1].imshow(unknown, cmap="gray")
    # for ax in axs.flat:
    #     ax.set_axis_off()
    # plt.show()

    # finally, apply the watershed algorithm with the foreground as seed. Before it, we
    # forcefully set the labels for all noncorneal pixels to be the same - this helps
    # with small edges around, e.g., reflections. As usual, we also set the unknown
    # regions to zero
    _, labels = cv.connectedComponents(foreground)
    labels = labels + 1
    labels[unknown > 0] = 0
    assert corneal_mask[0, 0] == 0, "top-left pixel belongs to the cornea!"
    labels[corneal_mask == 0] = labels[0, 0]
    markers = cv.watershed(img[..., :3], labels)  # remove alpha channel if present
    # segmentation = img[..., :3].copy()
    # segmentation[markers == -1] = [255, 0, 0]  # boundaries are marked with -1
    # plt.imshow(segmentation)
    # plt.axis("off")
    # plt.show()

    # now that the watershed segmentation is available, we need to extract the contours
    # (i.e., boundaries) of all the nonzero regions. We find the contours of the
    # segmented regions via thresholding + findContours (see
    # https://stackoverflow.com/a/50889494/19648688), but we additionally take care to
    # remove contours that extend outside the cornea
    _, contours_mask = cv.threshold(
        markers.astype(np.uint8), 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU
    )

    # finally, find all the contours (hierarchy information is disregarded)
    corneal_contours_mask = cv.bitwise_and(contours_mask, corneal_mask)
    tb_contours, _ = cv.findContours(
        corneal_contours_mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE
    )

    # create a mask by filling all found contours, thus highlighting all the pixels that
    # we believe to be positive to the TB stain. Just the make sure, set once again to
    # zero all the pixels that are outside the cornea
    tb_positive_mask = np.zeros(img.shape[:2], np.uint8)
    for i in range(len(tb_contours)):
        cv.drawContours(tb_positive_mask, tb_contours, i, 255, cv.FILLED)
    tb_positive_mask = cv.bitwise_and(tb_positive_mask, corneal_mask)
    # img[tb_positive_mask > 0] = (255, 0, 0, 255) if has_four_channels else (255, 0, 0)
    # plt.imshow(cv.cvtColor(img, cv.COLOR_BGRA2RGBA))
    # plt.axis("off")
    # plt.show()
    return corneal_mask, tb_positive_mask, tb_contours


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
    has_four_channels = img.shape[2] == 4
    if has_four_channels:
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


def calculate_mortality_per_circle(
    corneal_mask: np.ndarray,
    tb_positive_mask: np.ndarray,
    center: np.ndarray,
    radius: int,
) -> tuple[int, int]:
    """Calculates the mortality of the cells within the circle defined by the given
    center and radius.

    Parameters
    ----------
    corneal_mask : np.ndarray
        The mask of the corneal pixel in the original image.
    tb_positive_mask : np.ndarray
        The mask of the TB-positive corneal  pixels in the original image.
    center : 2d array of ints
        center of the circle for which the mortality is calculated.
    radius : float
        radius of the circle for which the mortality is calculated.

    Returns
    -------
    tuple of 2 int
        The mortality of the cells within the specified circle as the number of
        TB-positive pixels and the number of all pixels. The mortality can be calculated
        as the ratio of the two.
    """
    mask = np.zeros(img.shape[:2], np.uint8)
    cv.circle(mask, center, radius, 255, cv.FILLED)
    corneal_submask = cv.bitwise_and(mask, corneal_mask)
    tb_positive_submask = cv.bitwise_and(mask, tb_positive_mask)
    dead = cv.countNonZero(tb_positive_submask)
    all = cv.countNonZero(corneal_submask)
    return dead, all


if __name__ == "__main__":
    args = parse_args()

    # read image
    img_path = args.img
    img = cv.imread(img_path, cv.IMREAD_UNCHANGED)  # BGR or BGRA
    if img is None:
        print("Could not open or find the image:", img_path)
        exit(1)

    # pre-process image
    img = preprocess(img, args.rotate)

    # convert to grayscale and segment
    gray_img = cv.cvtColor(
        img, cv.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv.COLOR_BGR2GRAY
    )
    corneal_mask, tb_positive_mask, tb_contours = find_contours_TB_pixels(
        img,
        gray_img,
        args.clahe,
        args.adaptive_thres_size,
        args.adaptive_thres_const,
    )

    # calculate mortality per enclosing circles
    num_circles = 5
    center, r = calculate_enclosing_circle(img, gray_img)
    center = center.astype(int)
    mortalities = [
        calculate_mortality_per_circle(
            corneal_mask, tb_positive_mask, center, int(r / num_circles * frac)
        )
        for frac in range(1, num_circles + 1)
    ]

    # print mortality data
    filelines = ["ring,dead,all,mortality"]
    for i, (dead, all) in enumerate(mortalities, start=1):
        filelines.append(f"{i}/{num_circles},{dead},{all},{dead / all}")
    dead_all, all = cv.countNonZero(tb_positive_mask), cv.countNonZero(corneal_mask)
    filelines.append(f"whole,{dead_all},{all},{dead_all / all}")
    filetext = "\n".join(filelines)
    if not args.yes:
        print(filetext)

    # create segmented image and plot it
    tb_color = [0, 0, 0]
    ring_color = [255, 0, 0]
    if img.shape[2] == 4:
        tb_color.append(255)
        ring_color.append(255)
    thickness = min(img.shape[:2]) * 3 // 1000
    for i in range(len(tb_contours)):
        cv.drawContours(img, tb_contours, i, tb_color, thickness * 2 // 3, cv.LINE_AA)

    for frac in range(1, num_circles + 1):
        cv.circle(
            img, center, int(r / num_circles * frac), ring_color, thickness, cv.LINE_AA
        )
    cv.circle(img, center, int(r / 100), ring_color, cv.FILLED, cv.LINE_AA)
    cv.putText(
        img,
        "center",
        np.subtract(center, (r / 10, r / 20)).astype(int),
        cv.FONT_HERSHEY_SIMPLEX,
        thickness * 0.5,
        ring_color,
        thickness,
        cv.LINE_AA,
    )
    if not args.yes:
        import matplotlib.pyplot as plt

        plt.imshow(cv.cvtColor(img, cv.COLOR_BGRA2RGBA))
        plt.axis("off")
        plt.show()

    # save segmented image and mortality data
    r = args.rotate
    new_path = img_path.with_stem(f"{img_path.stem}{r} (segmented)")
    cv.imwrite(new_path, img)
    with open(img_path.with_name(f"{img_path.stem}{r} (mortalities).csv"), "w") as file:
        file.writelines(filetext)
