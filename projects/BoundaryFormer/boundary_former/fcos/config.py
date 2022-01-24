import os

from detectron2.config import CfgNode as CN

def add_fcos_config(cfg):
    """
    Add config for FCOS.
    """
    cfg.MODEL.FCOS = CN()

    # This is the number of foreground classes.
    cfg.MODEL.FCOS.NUM_CLASSES = 80
    cfg.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    cfg.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
    cfg.MODEL.FCOS.PRIOR_PROB = 0.01
    cfg.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
    cfg.MODEL.FCOS.NMS_TH = 0.6
    cfg.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 1000
    cfg.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000
    cfg.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
    cfg.MODEL.FCOS.POST_NMS_TOPK_TEST = 100
    cfg.MODEL.FCOS.TOP_LEVELS = 2
    cfg.MODEL.FCOS.NORM = "GN"  # Support GN or none
    cfg.MODEL.FCOS.USE_SCALE = True

    # Multiply centerness before threshold
    # This will affect the final performance by about 0.05 AP but save some time
    cfg.MODEL.FCOS.THRESH_WITH_CTR = False
    
    # Focal loss parameters
    cfg.MODEL.FCOS.LOSS_ALPHA = 0.25
    cfg.MODEL.FCOS.LOSS_GAMMA = 2.0
    cfg.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]
    cfg.MODEL.FCOS.USE_RELU = True
    cfg.MODEL.FCOS.USE_DEFORMABLE = False
    
    # the number of convolutions used in the cls and bbox tower
    cfg.MODEL.FCOS.NUM_CLS_CONVS = 4
    cfg.MODEL.FCOS.NUM_BOX_CONVS = 4
    cfg.MODEL.FCOS.NUM_SHARE_CONVS = 0
    cfg.MODEL.FCOS.CENTER_SAMPLE = True
    cfg.MODEL.FCOS.POS_RADIUS = 1.5
    cfg.MODEL.FCOS.LOC_LOSS_TYPE = 'giou'
    
