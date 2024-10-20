import argparse
from pathlib import Path

import cv2 as cv
import numpy as np
import skimage.metrics as skim
import sklearn.metrics as sklm


def adjust_masks(img: np.ndarray) -> np.ndarray:
    """Makes sure that the background class is False and the TB-positive pixels are
    True.

    Parameters
    ----------
    img : np.ndarray
        The image to adjust.

    Returns
    -------
    np.ndarray
        The adjusted boolean image.
    """
    h, w = img.shape
    corners = ((0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1))
    bg_candidates = {img[y, x] for x, y in corners}
    if len(bg_candidates) != 1:
        raise ValueError("could not determine background class from corner pixels")
    bg = bg_candidates.pop()
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[img != bg] = 1
    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compares segmentation results with ground truth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pred", type=Path, help="Predicted segmentation filepath")
    parser.add_argument("targ", type=Path, help="Target segmentation filepath")
    args = parser.parse_args()

    # read images
    pred_path = args.pred
    pred = cv.imread(pred_path, cv.IMREAD_GRAYSCALE)
    targ_path = args.targ
    targ = cv.imread(targ_path, cv.IMREAD_GRAYSCALE)
    for img, path in ((pred, pred_path), (targ, targ_path)):
        if img is None:
            print("Could not open or find the image:", path)
            exit(1)

    # adjust classes - some segmentations are white-in-black, others are black-in-white
    norm = 100 / min(pred.shape)
    pred = adjust_masks(pred)
    targ = adjust_masks(targ)
    fpred = pred.flatten()
    ftarg = targ.flatten()

    # define the metrics to compute
    precision, recall, f1, _ = sklm.precision_recall_fscore_support(
        ftarg, fpred, average=None
    )
    metrics = {
        "accuracy": sklm.accuracy_score(ftarg, fpred),
        "balanced-accuracy": sklm.balanced_accuracy_score(ftarg, fpred),
        "precision": precision,
        "recall": recall,
        "dice": f1,
        "iou": sklm.jaccard_score(ftarg, fpred, average=None),
        "standard-hausdorff": skim.hausdorff_distance(targ, pred, "standard") * norm,
        "modified-hausdorff": skim.hausdorff_distance(targ, pred, "modified") * norm,
    }

    # create CSV file with metrics
    filelines = ["class," + ",".join(metrics.keys())]
    bg_metrics, fg_metrics = [], []
    for metric in metrics.values():
        if isinstance(metric, (float, int)):
            bg_metrics.append("-")
            fg_metrics.append(metric)
        else:
            bg_metrics.append(metric[0])
            fg_metrics.append(metric[1])
    filelines.append("background," + ",".join(map(str, bg_metrics)))
    filelines.append("foreground," + ",".join(map(str, fg_metrics)))
    filetext = "\n".join(filelines)

    # save metrics to disk
    new_path = pred_path.with_name(f"{pred_path.stem}-vs-{targ_path.stem}-metrics.csv")
    with open(new_path, "w") as file:
        file.writelines(filetext)
