# AI4VCHECK

[![Source Code License](https://img.shields.io/badge/license-GPL-blueviolet)](https://github.com/FilippoAiraldi/AI4VCHECK/blob/master/LICENSE)
![Python 3.13.2](https://img.shields.io/badge/python-3.13.2-green.svg)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://docs.astral.sh/ruff/)

## Segmentation

In order to run the segmentation, follow the steos below:

1. create a directory containing the image you want to segment
1. upload the image to [Segment Anything](https://segment-anything.com/demo#) and use it to extract just the cornea from the whole image. Save the corneal image to the same directory
1. create an `args.json` in the same directory with the following content:

    ```json
    {
        "clahe": false,
        "adaptive_thres_size": 101,
        "adaptive_thres_const": 3.0
    }
    ```

    These value can be adjusted to your segmentation needs.

1. run the segmentation script:

    ```bash
    python3 segment.py path/to/corneal/image.png
    ```

    The script will automatically pick the arguments from the `args.json` file and will save the segmented image and mortality rates per ring in the same directory.
