import math
import numpy as np
from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, cat
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

@SEM_SEG_HEADS_REGISTRY.register()
class BoundaryFormerSemSegHead(nn.Module):
    """
    A semantic segmentation head that combines a head set in `POINT_HEAD.COARSE_SEM_SEG_HEAD_NAME`
    and a point head set in `MODEL.POINT_HEAD.NAME`.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        self.num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES        

        # adopt a (very very) coarse semantic head to serve as a sort of semantic RPN.
        # 1. According to some downsampled-resolution, take the top ranked classes across
        #    all spatial positions (possibly multiple for a single cell).
        # 2. Map these positions to image-space boxes and initialize a polygon within.
        # 3. Supervise as the union of the rasterized masks.
        if len(cfg.MODEL.BOUNDARY_HEAD.COARSE_SEM_SEG_HEAD_NAME) > 0:
            self.coarse_sem_seg_head = SEM_SEG_HEADS_REGISTRY.get(
                cfg.MODEL.BOUNDARY_HEAD.COARSE_SEM_SEG_HEAD_NAME
            )(cfg, input_shape)
        else:
            self.coarse_sem_seg_head = None
            

    def forward(self, features, images, targets=None):
        if self.coarse_sem_seg_head is None:
            if not self.training:
                raise ValueError("not supported")

            # use the p6 shape and find the ratio <= that is divisible.
            target_shape = targets.shape[2:]
            coarse_shape = features["p6"].shape[2:]
            #stride_factor = (math.ceil(target_shape[0] / coarse_shape[0]), math.ceil(

            targets_shape = targets.shape[1:]
            if ((targets_shape[0] % self.coarsifying_rate) != 0) or ((targets_shape[1] % self.coarsifying_rate) != 0):
                # I don't think this is guaranteed at all.
                print("Not divisible")
                import pdb
                pdb.set_trace()

            import pdb
            pdb.set_trace()
                
            targets[targets == self.ignore_value] = self.num_classes
            coarsified = self.coarsify(F.one_hot(targets, num_classes=self.num_classes + 1))[:, :self.num_classes]            
        else:
            # todo, predict this from
            raise ValueError("not supported")
        
