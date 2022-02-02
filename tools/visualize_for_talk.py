#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
from itertools import chain
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.structures.boxes import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, random_color

def create_text_labels(classes, scores, class_names, is_crowd=None):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):
        is_crowd (list[bool] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None:
        if class_names is not None and len(class_names) > 0:
            labels = [class_names[i] for i in classes]
        else:
            labels = [str(i) for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    if labels is not None and is_crowd is not None:
        labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
    return labels

if __name__ == "__main__":
    dirname = "for_talk"
    os.makedirs(dirname, exist_ok=True)
    metadata = MetadataCatalog.get("coco_2017_val")

    def output(vis, fname):
        filepath = os.path.join(dirname, fname)
        print("Saving to {} ...".format(filepath))
        vis.save(filepath)

    scale = 1.0
    TEASER_IMAGE_ID = 7816
    VAL_DICTS = list(DatasetCatalog.get("coco_2017_val"))
    TEASER_DICT = [d for d in VAL_DICTS if d["image_id"] == TEASER_IMAGE_ID][0]

    def write_teaser():
        annos = TEASER_DICT["annotations"]
        num_instances = len(annos)
        assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]
        
        img = utils.read_image(TEASER_DICT["file_name"], "RGB")
        visualizer = Visualizer(img, metadata=metadata, scale=scale)

        boxes = [
            BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS)
            if len(x["bbox"]) == 4
            else x["bbox"]
            for x in annos
        ]

        colors = None
        category_ids = [x["category_id"] for x in annos]
        colors = [
            visualizer._jitter([x / 255 for x in metadata.thing_colors[c]])
            for c in category_ids
        ]

        names = metadata.get("thing_classes", None)
        labels = create_text_labels(
            category_ids,
            scores=None,
            class_names=names,
            is_crowd=[x.get("iscrowd", 0) for x in annos],
        )

        vis = visualizer.overlay_instances(
            labels=labels, boxes=boxes, assigned_colors=colors)
        output(vis, "teaser_boxes.png")

    write_teaser()
