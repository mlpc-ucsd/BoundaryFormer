import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json

logger = logging.getLogger(__name__)

KINS_CATEGORIES = [
    {"color": [119, 11, 32], "isthing": 1, "id": 1, "name": "cyclist"},
    {"color": [220, 20, 60], "isthing": 1, "id": 2, "name": "pedestrian"},
    {"color": [0, 0, 230], "isthing": 1, "id": 3, "name": "rider"},
    {"color": [0, 0, 142], "isthing": 1, "id": 4, "name": "car"},
    {"color": [0, 80, 100], "isthing": 1, "id": 5, "name": "tram"},
    {"color": [0, 0, 70], "isthing": 1, "id": 6, "name": "truck"},
    {"color": [0, 60, 100], "isthing": 1, "id": 7, "name": "van"},
    {"color": [220, 220, 0], "isthing": 1, "id": 8, "name": "misc"},
]

def _get_kins_instances_meta():
    thing_ids = [k["id"] for k in KINS_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in KINS_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 8, len(thing_ids)
    # Mapping from the incontiguous KINS category id to an id in [0, 7]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in KINS_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def get_load_fn(json_file, image_dir, name, box_in_key, segmentation_in_key):
    return lambda: load_coco_json(json_file, image_dir, name, box_in_key=box_in_key, segmentation_in_key=segmentation_in_key)

def register_all_kins(root):
    root = os.path.join(root, "kitti")
    meta = _get_kins_instances_meta()
    
    for name, image_dirname, annotations_path in [
        ("train", "train", "annotations/train_2020.json"),
        ("val", "val", "annotations/val_2020.json")
    ]:
        for kind in ["inmodal", "amodal"]:
            is_amodal = "amodal" in name
            if is_amodal:
                print("Loading amodal keys")
                box_in_key = "a_bbox"
                segmentation_in_key = "a_segm"
            else:
                box_in_key = "i_bbox"
                segmentation_in_key = "i_segm"

            dataset_name = f"kins_2020_{kind}_instance_seg_{name}"                
            image_dir = os.path.join(root, image_dirname)
            json_file = os.path.join(root, annotations_path)
            
            DatasetCatalog.register(dataset_name, get_load_fn(json_file, image_dir, name, box_in_key=box_in_key, segmentation_in_key=segmentation_in_key))
            MetadataCatalog.get(dataset_name).set(
                json_file=json_file, image_root=image_dir, evaluator_type="coco", **_get_kins_instances_meta()
            )            

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_kins(_root)
