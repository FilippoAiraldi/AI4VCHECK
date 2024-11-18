import argparse
import json
from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np

from circles import N_CIRCLES, calculate_enclosing_circle


def preprocess(img: np.ndarray, angle: float) -> np.ndarray:
    """Pre-processes the image prior to segmentation.

    Parameters
    ----------
    img : np.ndarray
        A 3- or 4-channel image containing the cornea. If the image has 4 channels,
        areas outside of the cornea should be transparent. If the image has 3 channels,
        areas outside of the cornea should be black.
    angle : float
        Rotation of the original image in degrees.

    Returns
    -------
    np.ndarray
        The pre-processed image.
    """
    if angle == 0.0:
        return img
    rows, cols = img.shape[:2]
    center = (cols / 2, rows / 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    new_rows = int(rows * abs_sin + cols * abs_cos)
    new_cols = int(rows * abs_cos + cols * abs_sin)
    M[0, 2] += new_rows / 2 - center[0]
    M[1, 2] += new_cols / 2 - center[1]
    return cv.warpAffine(img, M, (new_rows, new_cols))


def find_contours_TB_pixels(
    img: np.ndarray,
    gray_img: Optional[np.ndarray] = None,
    clahe: bool = False,
    adaptive_thres_size: int = 101,
    adaptive_thres_const: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...], np.ndarray]:
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
    corneal mask, and TB-positive mask, and contours + hierarchy
        The mask of the corneal pixeles (as an array), the TB-positive mask (as an
        array) of the regions that are believed to be stained by the TB dye, the
        watershed-segmented contours of these regions (as a tuple of arrays) and their
        hierarchy.
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
    cv.imwrite("Figure Y (2).png", blurred_img)
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
    cv.imwrite("Figure Y (2).png", mask)

    # then, for each nonzero pixel compute the distance to the nearest zero pixel, and
    # use it to discern foreground, background and unknown pixels
    distance = cv.distanceTransform(mask, cv.DIST_L2, cv.DIST_MASK_3)
    _, foreground = cv.threshold(
        distance,
        0.1 * distance.max(),  # NOTE: tunable (!!!) - threshold multiplier
        255,
        cv.THRESH_BINARY,
    )
    # cv.normalize(distance, distance, 0, 255.0, cv.NORM_MINMAX)
    cv.imwrite("Figure Y (3).png", distance)
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
    cv.imwrite("Figure Y (4).png", foreground)

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
    # import matplotlib.pyplot as plt
    # cmap = plt.get_cmap("twilight")
    # cmapped_img = cmap((markers - markers.min()) / np.ptp(markers)) * 255
    # cmapped_img[(markers == markers[0, 0]) | (markers == markers[1, 1])] = [0, 0, 0, 0]
    # OR
    cmapped_img = np.zeros((*img.shape[:2], 4), np.uint8)
    rng = np.random.default_rng(0)
    for marker in np.unique(markers):
        if marker not in (-1, 0, 1):
            cmapped_img[markers == marker] = [*rng.integers(0, 256, 3), 255]
    cv.imwrite("Figure Y (5).png", cmapped_img)

    # now that the watershed segmentation is available, we need to extract the contours
    # (i.e., boundaries) of all the nonzero regions. We find the contours of the
    # segmented regions via thresholding + findContours (see
    # https://stackoverflow.com/a/50889494/19648688), but we additionally take care to
    # remove contours that extend outside the cornea
    _, contours_mask = cv.threshold(
        markers.astype(np.uint8), 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU
    )

    # finally, find all the contours
    corneal_contours_mask = cv.bitwise_and(contours_mask, corneal_mask)
    tb_contours, tb_hierarchy = cv.findContours(
        corneal_contours_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE
    )
    assert tb_hierarchy.shape[0] == 1, "more than one hierarchy found!"

    # create a mask by filling all found contours, thus highlighting all the pixels that
    # we believe to be positive to the TB stain. Just the make sure, set once again to
    # zero all the pixels that are outside the cornea
    tb_positive_mask = np.zeros(img.shape[:2], np.uint8)
    for i in range(len(tb_contours)):
        if tb_hierarchy[0, i, 3] == -1:
            cv.drawContours(
                tb_positive_mask, tb_contours, i, 255, cv.FILLED, hierarchy=tb_hierarchy, maxLevel=1
            )
    tb_positive_mask = cv.bitwise_and(tb_positive_mask, corneal_mask)
    cv.imwrite("Figure Y (6).png", tb_positive_mask)
    # img[tb_positive_mask > 0] = (255, 0, 0, 255) if has_four_channels else (255, 0, 0)
    # plt.imshow(cv.cvtColor(img, cv.COLOR_BGRA2RGBA))
    # plt.axis("off")
    # plt.show()
    return corneal_mask, tb_positive_mask, tb_contours


def calculate_enclosing_ellipse(
    img: np.ndarray, gray_img: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray, float]:
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
    center, diameters, and angle
        The center, diameters and rotation angle (deg) of the ellipse that best
        approximates the corneal segmented image.
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

    # fit best ellipse to the contour
    ellipse = cv.fitEllipse(cnt)  # (center, (major diam., minor diam.), clock. angle)
    return np.asarray(ellipse[0]), np.asarray(ellipse[1]), ellipse[2]


def calculate_mortality_per_ellipse(
    corneal_mask: np.ndarray,
    tb_positive_mask: np.ndarray,
    center: np.ndarray,
    diameters: np.ndarray,
    angle: float,
) -> tuple[int, int]:
    """Calculates the mortality of the cells within the ellipse defined by the given
    center, diameters and rotation angle.

    Parameters
    ----------
    corneal_mask : np.ndarray
        The mask of the corneal pixel in the original image.
    tb_positive_mask : np.ndarray
        The mask of the TB-positive corneal  pixels in the original image.
    center : 2d array of ints
        Center of the ellipse for which the mortality is calculated.
    diameters : float
        Diameters of the ellipse for which the mortality is calculated.
    angle : float
        Rotation angle of the ellipse for which the mortality is calculated.

    Returns
    -------
    tuple of 2 int
        The mortality of the cells within the specified ellipse as the number of
        TB-positive pixels and the number of all pixels. The mortality can be calculated
        as the ratio of the two. The third returned element is the mask of the
        TB-positive pixels in that circle.
    """
    mask = np.zeros(img.shape[:2], np.uint8)
    cv.ellipse(mask, (center, diameters, angle), 255, cv.FILLED)
    corneal_submask = cv.bitwise_and(mask, corneal_mask)
    tb_positive_submask = cv.bitwise_and(mask, tb_positive_mask)
    dead = cv.countNonZero(tb_positive_submask)
    all = cv.countNonZero(corneal_submask)
    return dead, all, tb_positive_submask


if __name__ == "__main__":
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        description="VCHECK segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("img", type=Path, help="Filepath of image to segment")
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
    group = parser.add_argument_group("Storing options")
    group.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prior to saving to disk",
    )
    group.add_argument(
        "--segmentation-folder",
        type=str,
        default="segmented",
        help="Folder where to store the segmented image",
    )
    group.add_argument(
        "--mortality-folder",
        type=str,
        default="mortality",
        help="Folder where to store the mortality data",
    )
    group.add_argument(
        "--save-masks",
        choices=[*map(str, range(1, N_CIRCLES + 1)), "whole"],
        default=[],
        nargs="*",
        type=str,
        help="Circles (or whole image) for which to save the masks",
    )
    group.add_argument(
        "--mask-folder",
        type=str,
        default="masks",
        help="Folder where to store the mask results",
    )
    args = parser.parse_args()

    # load arguments from json - if not provided, see if there is a file named args.json
    if not args.args_json:
        args_json = args.img.with_name("args.json")
        if args_json.is_file():
            args.args_json = args_json
    if args.args_json:
        with open(args.args_json, "r") as file:
            data = json.load(file)
            for key, value in data.items():
                setattr(args, key, value)

    # read image
    path = args.img
    img = cv.imread(path, cv.IMREAD_UNCHANGED)  # BGR or BGRA
    if img is None:
        print("Could not open or find the image:", path)
        exit(1)

    # pre-process image
    rot = args.rotate
    img = preprocess(img, rot)

    # convert to grayscale and segment
    gray_img = cv.cvtColor(
        img, cv.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv.COLOR_BGR2GRAY
    )
    corneal_mask, tb_positive_mask, tb_contours, tb_hierarchy = find_contours_TB_pixels(
        img,
        gray_img,
        args.clahe,
        args.adaptive_thres_size,
        args.adaptive_thres_const,
    )

    # calculate mortality per enclosing ellipses
    num_steps = 5
    center, diameters, angle = calculate_enclosing_ellipse(img, gray_img)
    mortalities = [
        calculate_mortality_per_ellipse(
            corneal_mask, tb_positive_mask, center, diameters / num_steps * frac, angle
        )
        for frac in range(1, num_steps + 1)
    ]

    # print mortality data
    filelines = ["ring,dead,all,mortality"]
    for i, (dead, all) in enumerate(mortalities, start=1):
        filelines.append(f"{i}/{num_steps},{dead},{all},{dead / all}")
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
        if tb_hierarchy[0, i, 3] == -1:
            cv.drawContours(
                img,
                tb_contours,
                i,
                tb_color,
                -thickness * 2 // 3,
                cv.LINE_AA,
                hierarchy=tb_hierarchy,
            )

    for frac in range(1, num_steps + 1):
        cv.ellipse(
            img,
            (center, diameters / num_steps * frac, angle),
            ring_color,
            thickness,
            cv.LINE_AA,
        )
    cv.circle(
        img,
        center.astype(int),
        int(diameters.mean() / 100),
        ring_color,
        cv.FILLED,
        cv.LINE_AA,
    )
    cv.putText(
        img,
        "center",
        np.subtract(center, (diameters[0] / 10, diameters[1] / 20)).astype(int),
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

    # save segmented image, mortality data and masks
    segmentation_folder = path.parent / args.segmentation_folder
    segmentation_folder.mkdir(exist_ok=True)
    new_path = segmentation_folder / f"{path.stem}-segmented-at-{rot}{path.suffix}"
    cv.imwrite(new_path, img)

    mortality_folder = path.parent / args.mortality_folder
    mortality_folder.mkdir(exist_ok=True)
    new_path = mortality_folder / f"{path.stem}-segmented-at-{rot}.csv"
    with open(new_path, "w") as file:
        file.writelines(filetext)

    if args.save_masks:
        mask_folder = path.parent / args.mask_folder
        mask_folder.mkdir(exist_ok=True)
        for circle in args.save_masks:
            mask = mortality_data[circle if circle == "whole" else int(circle)][2]
            new_path = mask_folder / f"{path.stem}-mask-{circle}-at-{rot}{path.suffix}"
            cv.imwrite(new_path, mask)
