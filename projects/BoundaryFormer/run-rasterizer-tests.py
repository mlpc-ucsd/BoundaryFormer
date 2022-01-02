import itertools
import numpy as np
import tqdm

import torch
from torch.autograd import Variable
from torch import nn

from detectron2.data import DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.structures.masks import rasterize_polygons_within_box

from boundary_former.diff_ras import dice_loss
from boundary_former.layers.diff_ras.polygon import SoftPolygon, SoftPolygonPyTorch
from boundary_former.utils import sample_ellipse_fast, pad_polygons, clip_and_normalize_polygons

import pdb

DEVICE = torch.device("cuda:0")
DATASET_NAME = "coco_2017_val"
NUMBER_VERTEX = 128
RESOLUTION = 128
LOSS_FN = dice_loss
THRESHOLD = 0.9987

INV_SMOOTHNESS = [1.0] #[10.0, 1.0, 0.1, 0.001, 0.0001]
HARD_CUDA_RASTERIZER = SoftPolygon(mode="hard_mask")
CUDA_RASTERIZERS = [SoftPolygon(mode="mask", inv_smoothness=inv_smoothness) for inv_smoothness in INV_SMOOTHNESS]
PYTORCH_RASTERIZERS = [SoftPolygonPyTorch(inv_smoothness=inv_smoothness) for inv_smoothness in INV_SMOOTHNESS]

def is_simple_polygon(ann):
    return isinstance(ann["segmentation"], list) and len(ann["segmentation"]) == 1

def rasterize_instances(rasterizer, segmentations, side_length, offset=-0.50):
    all_polygons = clip_and_normalize_polygons(torch.from_numpy(pad_polygons(list(itertools.chain.from_iterable([
        [np.array(segmentation[0]).reshape(-1, 2) for segmentation in segmentations]])))).float().to(DEVICE))

    # to me it seems the offset would need to be in _pixel_ space?
    return rasterizer(all_polygons * float(side_length) + offset, side_length, side_length, 1.0)    

agreement_percentages = []
for dict in tqdm.tqdm(DatasetCatalog.get(DATASET_NAME)):
    annotations = [ann for ann in dict["annotations"] if is_simple_polygon(ann)]
    if len(annotations) == 0:
        continue

    # only keep annotations which are a single polygon (and not masks)
    num_instances = len(annotations)

    boxes = np.asarray([ann["bbox"] for ann in annotations]).reshape(-1, 4)
    xyxys = BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    
    segmentations = [ann["segmentation"] for ann in annotations]
    ground_truth_masks = torch.stack([rasterize_polygons_within_box([np.array(segmentation[0])], xyxy, RESOLUTION) for xyxy, segmentation in zip(xyxys, segmentations)]).float().to(DEVICE)
    ground_truth_rasterized = rasterize_instances(HARD_CUDA_RASTERIZER, segmentations, RESOLUTION)
    agreement_percentage = torch.count_nonzero(ground_truth_masks == ground_truth_rasterized) / float(ground_truth_masks.numel())

    agreement_percentages.append(agreement_percentage.item())
    print("GT Rasterization agreement: {0}".format(np.mean(agreement_percentages)))
    
    # check whether if we rasterize the GT, we get the GT (subject to some adjustment).

    # check the gradients are good.
    predictions = Variable(sample_ellipse_fast(
        RESOLUTION * 0.5 * torch.ones((1, num_instances), device=DEVICE),
        RESOLUTION * 0.5 * torch.ones((1, num_instances), device=DEVICE),
        RESOLUTION * 0.49 * torch.ones((1, num_instances), device=DEVICE),
        RESOLUTION * 0.49 * torch.ones((1, num_instances), device=DEVICE),
        count=NUMBER_VERTEX).view(num_instances, NUMBER_VERTEX, 2).float())
    predictions.requires_grad = True

    for rasterizer_idx, (cuda_rasterizer, pytorch_rasterizer) in enumerate(zip(CUDA_RASTERIZERS, PYTORCH_RASTERIZERS)):
        tau = INV_SMOOTHNESS[rasterizer_idx]
        if predictions.grad is not None:
            predictions.grad.data.zero_()
            
        rasterized_cuda = cuda_rasterizer(predictions - 0.5, RESOLUTION, RESOLUTION, 1.0)
        loss_value_cuda = LOSS_FN(rasterized_cuda, ground_truth_masks)
        loss_value_cuda.backward()
        grad_cuda = predictions.grad.data.cpu().clone()

        predictions.grad.data.zero_()
        rasterized_pytorch = pytorch_rasterizer(predictions - 0.5, RESOLUTION, RESOLUTION, 1.0)
        loss_value_pytorch = LOSS_FN(rasterized_pytorch, ground_truth_masks)
        loss_value_pytorch.backward()
        grad_pytorch = predictions.grad.data.cpu().clone()

        is_rasterized_close = torch.isclose(rasterized_cuda, rasterized_pytorch)
        rasterized_agreement_percentage = torch.count_nonzero(is_rasterized_close) / float(is_rasterized_close.numel())

        print("Rasterized agreement (tau = {0}): {1}".format(tau, rasterized_agreement_percentage))
        if rasterized_agreement_percentage < THRESHOLD:
            import pdb
            pdb.set_trace()    
        
        is_gradient_close = torch.isclose(grad_cuda, grad_pytorch)
        gradient_agreement_percentage = torch.count_nonzero(is_gradient_close) / float(is_gradient_close.numel())
        
        print("Gradient agreement (tau = {0}): {1}".format(tau, gradient_agreement_percentage))
        if gradient_agreement_percentage < THRESHOLD:
            import pdb
            pdb.set_trace()    
