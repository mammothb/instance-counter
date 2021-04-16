import numpy as np

import src.mrcnn.visualize as mrcnn_viz

def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
        A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label, label_merge):
    """Adds color defined by the dataset colormap to the label.

    Args:
        label: A 2D array with integer type, storing the segmentation label.

    Returns:
        result: A 2D array with floating type. The element of the array
            is the color indexed by the corresponding element in the input label
            to the PASCAL color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
            map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError("Expect 2-D input label")

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError("label value too large.")

    return colormap[label_merge[label]]


def display_segmentation(image, seg_map, label_names, label_merge):
    """Visualizes input image, segmentation map and overlay view."""
    colormap = create_pascal_label_colormap()

    im_copy = image.copy()
    unique_labels = np.unique(seg_map)
    for i in unique_labels:
        if label_names[i] == "person":
            continue
        im_copy = mrcnn_viz.apply_mask(
            im_copy, seg_map == i, colormap[label_merge[i]][::-1]
        )

    return im_copy