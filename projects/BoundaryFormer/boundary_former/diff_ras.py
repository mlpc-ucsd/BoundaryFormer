import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.structures.instances import Instances
from detectron2.utils.events import get_event_storage

from boundary_former.layers.diff_ras.polygon import SoftPolygon
from boundary_former.utils import get_union_box, rasterize_instances, POLY_LOSS_REGISTRY

class ClippingStrategy(nn.Module):
    def __init__(self, cfg, is_boundary=False):
        super().__init__()

        self.register_buffer("laplacian", torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3))

        self.is_boundary = is_boundary
        self.side_lengths = np.array(cfg.MODEL.DIFFRAS.RESOLUTIONS).reshape(-1, 2)

    # not used.
    def _extract_target_boundary(self, masks, shape):
        boundary_targets = F.conv2d(masks.unsqueeze(1), self.laplacian, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        # odd? only if the width doesn't match?
        if boundary_targets.shape[-2:] != shape:
            boundary_targets = F.interpolate(
                boundary_targets, shape, mode='nearest')

        return boundary_targets

    def forward(self, instances, clip_boxes=None, lid=0):                
        device = self.laplacian.device

        gt_masks = []

        if clip_boxes is not None:
            clip_boxes = torch.split(clip_boxes, [len(inst) for inst in instances], dim=0)
            
        for idx, instances_per_image in enumerate(instances):
            if len(instances_per_image) == 0:
                continue

            if clip_boxes is not None:
                # todo, need to support rectangular boxes.
                gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                    clip_boxes[idx].detach(), self.side_lengths[lid][0])
            else:
                gt_masks_per_image = instances_per_image.gt_masks.rasterize_no_crop(self.side_length).to(device)

            # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
            gt_masks.append(gt_masks_per_image)

        return torch.cat(gt_masks).squeeze(1)

def dice_loss(input, target):
    smooth = 1.

    iflat = input.reshape(-1)
    tflat = target.reshape(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

@POLY_LOSS_REGISTRY.register()
class MaskRasterizationLoss(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()

        self.register_buffer("rasterize_at", torch.from_numpy(np.array(cfg.MODEL.DIFFRAS.RESOLUTIONS).reshape(-1, 2)))
        self.inv_smoothness_schedule = cfg.MODEL.DIFFRAS.INV_SMOOTHNESS_SCHED
        self.inv_smoothness = self.inv_smoothness_schedule[0]
        self.inv_smoothness_iter = cfg.MODEL.DIFFRAS.INV_SMOOTHNESS_STEPS
        self.inv_smoothness_idx = 0
        self.iter = 0

        # whether to invoke our own rasterizer in "hard" mode.
        self.use_rasterized_gt = cfg.MODEL.DIFFRAS.USE_RASTERIZED_GT
        
        self.pred_rasterizer = SoftPolygon(inv_smoothness=self.inv_smoothness, mode="mask")
        self.clip_to_proposal = not cfg.MODEL.ROI_HEADS.PROPOSAL_ONLY_GT
        self.predict_in_box_space = cfg.MODEL.BOUNDARY_HEAD.PRED_WITHIN_BOX
        
        if self.clip_to_proposal or not self.use_rasterized_gt:
            self.clipper = ClippingStrategy(cfg)
            self.gt_rasterizer = None
        else:
            self.gt_rasterizer = SoftPolygon(inv_smoothness=1.0, mode="mask")

        self.offset = 0.5
        self.loss_fn = dice_loss
        self.name = "mask"

    def _create_targets(self, instances, clip_boxes=None, lid=0):
        if self.clip_to_proposal or not self.use_rasterized_gt:
            targets = self.clipper(instances, clip_boxes=clip_boxes, lid=lid)            
        else:            
            targets = rasterize_instances(self.gt_rasterizer, instances, self.rasterize_at)
            
        return targets

    def forward(self, preds, targets, lid=0):
        if isinstance(targets, torch.Tensor):
            device = targets.device
        else:
            device = targets[0].gt_boxes.device
            
        batch_size = len(preds)
        storage = get_event_storage()

        resolution = self.rasterize_at[lid]

        # advance to the right schedule.
        # this is really bad, please clean up.
        if (self.iter == 0) and (storage.iter > 0):
            for idx, start_iter in enumerate(self.inv_smoothness_iter):
                if start_iter > storage.iter:
                    self.inv_smoothness_idx = idx # - 1???
                    self.inv_smoothness = self.inv_smoothness_schedule[self.inv_smoothness_idx]
                    self.pred_rasterizer.inv_smoothness = self.inv_smoothness
                    break

            if len(self.inv_smoothness_iter) and (storage.iter > self.inv_smoothness_iter[-1]):
                self.inv_smoothness_idx = len(self.inv_smoothness_iter)
                self.inv_smoothness = self.inv_smoothness_schedule[self.inv_smoothness_idx]
                self.pred_rasterizer.inv_smoothness = self.inv_smoothness

            self.iter = 1
        else:
            self.iter = 1

        if ((len(self.inv_smoothness_schedule) > (self.inv_smoothness_idx + 1)) and
            (storage.iter == self.inv_smoothness_iter[self.inv_smoothness_idx])):
            self.inv_smoothness = self.inv_smoothness_schedule[self.inv_smoothness_idx + 1]
            self.pred_rasterizer.inv_smoothness = self.inv_smoothness
            self.inv_smoothness_idx += 1
            self.iter = 1

        # -0.5 needed to align the rasterizer with COCO.
        pred_masks = self.pred_rasterizer(preds * float(resolution[1].item()) - self.offset, resolution[1].item(), resolution[0].item(), 1.0).unsqueeze(1)
        # add a little noise since depending on inv_smoothness/predictions, we can exactly predict 0 or 1.0        
        pred_masks = torch.clamp(pred_masks, 0.00001, 0.99999)

        # NOTE: this is only compatible when pred_box is not changed.
        
        if isinstance(targets, torch.Tensor):            
            target_masks = targets            
        else:
            # if not pooled, we can:
            # enforce a full image loss (unlikely to be efficient)
            # rasterize only within the union of the GT box and the predicted polygon.
            # rasterize only within the predicted polygon (like an RoI) and clip the GT to that.
            clip_boxes = None
            if self.predict_in_box_space:
                # keep preds the same, but rasterize proposal box as GT.
                clip_boxes = torch.cat([t.proposal_boxes.tensor for t in targets])
            
            target_masks = self._create_targets(targets, clip_boxes=clip_boxes, lid=lid)
            target_masks = target_masks.unsqueeze(1).float()
            
        return self.loss_fn(pred_masks, target_masks), target_masks
