import os

from detectron2.config import CfgNode as CN

def add_boundaryformer_config(cfg):
    """
    Add config for BoundaryFormer.
    """
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.    
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Color augmentatition from SSD paper for semantic segmentation model during training.
    cfg.INPUT.COLOR_AUG_SSD = False
    
    cfg.MODEL.BOUNDARY_HEAD = CN()
    cfg.MODEL.BOUNDARY_HEAD.MODEL_DIM = 256
    cfg.MODEL.BOUNDARY_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.BOUNDARY_HEAD.POLY_INIT = "ellipse"
    cfg.MODEL.BOUNDARY_HEAD.POLY_NUM_PTS = 64
    cfg.MODEL.BOUNDARY_HEAD.POLY_LOSS = CN()
    cfg.MODEL.BOUNDARY_HEAD.POLY_LOSS.NAMES = ["MaskRasterizationLoss"]
    cfg.MODEL.BOUNDARY_HEAD.POLY_LOSS.WS = [1.0]    
    cfg.MODEL.BOUNDARY_HEAD.ITER_REFINE = True
    cfg.MODEL.BOUNDARY_HEAD.UPSAMPLING = True
    cfg.MODEL.BOUNDARY_HEAD.UPSAMPLING_BASE_NUM_PTS = 8
    cfg.MODEL.BOUNDARY_HEAD.NUM_DEC_LAYERS = 4
    cfg.MODEL.BOUNDARY_HEAD.USE_CLS_TOKEN = False
    cfg.MODEL.BOUNDARY_HEAD.USE_P2P_ATTN = True
    cfg.MODEL.BOUNDARY_HEAD.CLS_AGNOSTIC_MASK = True
    cfg.MODEL.BOUNDARY_HEAD.PRED_WITHIN_BOX = False
    cfg.MODEL.BOUNDARY_HEAD.PREPOOL = True
    cfg.MODEL.BOUNDARY_HEAD.MAX_PROPOSALS_PER_IMAGE = 0
    cfg.MODEL.BOUNDARY_HEAD.DROPOUT = 0.0
    cfg.MODEL.BOUNDARY_HEAD.DEEP_SUPERVISION = True
    cfg.MODEL.BOUNDARY_HEAD.COARSE_SEM_SEG_HEAD_NAME = "" #SemSegFPNHead"

    cfg.MODEL.DIFFRAS = CN()
    cfg.MODEL.DIFFRAS.RESOLUTIONS = [64, 64] 
    cfg.MODEL.DIFFRAS.RASTERIZE_WITHIN_UNION = False
    cfg.MODEL.DIFFRAS.USE_RASTERIZED_GT = False
    cfg.MODEL.DIFFRAS.INV_SMOOTHNESS_SCHED = (0.001,) #(0.15, 0.005)
    cfg.MODEL.DIFFRAS.INV_SMOOTHNESS_STEPS = () #(50000.)

    cfg.MODEL.ROI_HEADS.PROPOSAL_ONLY_GT = False
    
    cfg.SOLVER.OPTIMIZER = "ADAM"
    cfg.SOLVER.BASE_LR = 1e-4
    cfg.SOLVER.WEIGHT_DECAY = 0.05
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0

    cfg.COMMENT = "NONE"
    cfg.OUTPUT_PREFIX = "outputs" if os.getenv("DETECTRON2_OUTPUTS") is None else os.getenv("DETECTRON2_OUTPUTS")
    cfg.TRAIN_SET_STR = ""
    cfg.CFG_FILE_STR = "default"
    cfg.OPT_STR = "default"
    
