import os.path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from six.moves import urllib

from src.deeplab.config import DL_LABEL_MERGE, DL_LABEL_NAMES
from src.deeplab.model import DeepLabModel
import src.deeplab.visualize as dl_viz
from src.mrcnn import utils
import src.mrcnn.coco as coco
from src.mrcnn.config import MRCNN_LABEL_NAMES
from src.mrcnn.model import MaskRCNN
import src.mrcnn.visualize as mrcnn_viz


ROOT_DIR = os.path.abspath(".")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = os.path.join(ROOT_DIR, "weights")

# DeepLab
DL_MODEL_PATH = os.path.join(MODEL_DIR, "deeplab_model.tar.gz")
DL_URL = (
    "http://download.tensorflow.org/models/"
    "deeplabv3_mnv2_ade20k_train_2018_12_03.tar.gz"
)

if not os.path.exists(DL_MODEL_PATH):
    print("Downloading DeepLab model...")
    urllib.request.urlretrieve(DL_URL, DL_MODEL_PATH)
    print("Download complete, loading DeepLab model...")

DL_MODEL = DeepLabModel(DL_MODEL_PATH)
print("DeepLab model loaded successfully!")

# MRCNN
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

MRCNN_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(MRCNN_MODEL_PATH):
    utils.download_trained_weights(MRCNN_MODEL_PATH)
config = InferenceConfig()
# Create model object in inference mode.
MRCNN_MODEL = MaskRCNN(mode="inference", model_dir=LOGS_DIR, config=config)

# Load weights trained on MS-COCO
MRCNN_MODEL.load_weights(MRCNN_MODEL_PATH, by_name=True)

VIDEO_PATH = os.path.join(ROOT_DIR, "videos", "People_sample_2.mp4")

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
output_path = f"{os.path.splitext(VIDEO_PATH)[0]}_processed.avi"

vid_writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc("M", "J", "P", "G"),
    fps,
    (
        round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    ),
)

sec = 0
while True:
    print(sec)
    cap.set(1, sec * fps)
    # Get frame from the video
    has_frame, frame = cap.read()

    # Stop the program if reached end of video
    if not has_frame:
        print("Done processing !!!")
        print(f"Output file is stored as {output_path}")
        break

    _, seg_map = DL_MODEL.run(Image.fromarray(frame[:, :, ::-1]))
    seg_map = cv2.resize(
        seg_map, frame.shape[:-1][::-1], interpolation=cv2.INTER_NEAREST
    )
    frame_copy = dl_viz.display_segmentation(
        frame, seg_map, DL_LABEL_NAMES, DL_LABEL_MERGE
    )

    results = MRCNN_MODEL.detect([frame[:, :, ::-1]])
    r = results[0]
    frame_copy = mrcnn_viz.display_instances(
        frame_copy,
        r["rois"],
        r["masks"],
        r["class_ids"],
        MRCNN_LABEL_NAMES,
        r["scores"],
    )

    # Write the frame with the detection boxes
    vid_writer.write(frame_copy.astype(np.uint8))
    sec += 1
cap.release()

# if __name__ == "__main__":
#     process_video(video_path)
