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


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class Processor:
    def __init__(self, logs_path, semantic_seg_path, instance_seg_path):
        self.semantic_seg = DeepLabModel(semantic_seg_path)

        instance_seg_config = InferenceConfig()
        self.instance_seg = MaskRCNN(
            mode="inference", model_dir=logs_path, config=instance_seg_config
        )
        self.instance_seg.load_weights(instance_seg_path, by_name=True)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        output_path = f"{os.path.splitext(video_path)[0]}_processed.avi"

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
                print("Done processing!")
                print(f"Output file is stored as {output_path}")
                break

            _, seg_map = self.semantic_seg.run(
                Image.fromarray(frame[:, :, ::-1])
            )
            seg_map = cv2.resize(
                seg_map, frame.shape[:-1][::-1], interpolation=cv2.INTER_NEAREST
            )
            frame_copy = dl_viz.display_segmentation(
                frame, seg_map, DL_LABEL_NAMES, DL_LABEL_MERGE
            )

            results = self.instance_seg.detect([frame[:, :, ::-1]])
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


if __name__ == "__main__":
    root_dir = os.path.abspath(".")
    logs_dir = os.path.join(root_dir, "logs")
    weights_dir = os.path.join(root_dir, "weights")

    dl_model_path = os.path.join(weights_dir, "deeplab_model.tar.gz")
    mrcnn_model_path = os.path.join(weights_dir, "mask_rcnn_coco.h5")
    video_path = os.path.join(root_dir, "videos", "People_sample_2.mp4")

    processor = Processor(logs_dir, dl_model_path, mrcnn_model_path)
    processor.process_video(video_path)
