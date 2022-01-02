import copy
import fvcore.nn.weight_init as weight_init
import imageio
import itertools
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import List

from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.modeling import ROI_MASK_HEAD_REGISTRY
from detectron2.structures import Boxes, Instances

from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from torch.nn.utils.rnn import pad_sequence

from boundary_former.diff_ras import MaskRasterizationLoss
from boundary_former.layers.deform_attn import MSDeformAttn
from boundary_former.poolers import MultiROIPooler
from boundary_former.position_encoding import build_position_encoding
from boundary_former.tensor import NestedTensor
from boundary_former.transformer import DeformableTransformerDecoder, DeformableTransformerControlLayer, MLP, point_encoding, UpsamplingDecoderLayer
from boundary_former.utils import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, inverse_sigmoid, sample_ellipse_fast, POLY_LOSS_REGISTRY, _get_clones

@ROI_MASK_HEAD_REGISTRY.register()    
class BoundaryFormerPolygonHead(nn.Module):
    @configurable
    def __init__(self, input_shape: ShapeSpec, in_features, vertex_loss_fns, vertex_loss_ws, ref_init="ellipse",
                 model_dim=256, number_control_points=64, number_layers=4, vis_period=0,
                 is_upsampling=True, iterative_refinement=False, use_cls_token=False, num_classes=80, cls_agnostic=False,
                 predict_in_box_space=False, prepool=True, dropout=0.0, deep_supervision=True, **kwargs):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
        """
        super().__init__()

        self.input_shape = input_shape
        self.in_features = in_features
        self.num_feature_levels = len(self.in_features)
        self.ref_init = ref_init

        self.batch_size_div = 16

        if not ref_init in ["ellipse", "random"]:
            raise ValueError("unknown ref_init {0}".format(ref_init))
        
        self.number_control_points = number_control_points
        self.model_dimension = model_dim
        self.is_upsampling = is_upsampling
        self.iterative_refinement = iterative_refinement or self.is_upsampling
        self.use_cls_token = use_cls_token
        self.num_classes = num_classes
        self.cls_agnostic = cls_agnostic
        self.vis_period = vis_period
        self.predict_in_box_space = predict_in_box_space
        self.prepool = prepool
        self.dropout = dropout
        self.deep_supervision = deep_supervision

        self.vertex_loss_fns = []
        for loss_fn in vertex_loss_fns:
            loss_fn_attr_name = "vertex_loss_fn_{0}".format(loss_fn.name)
            self.add_module(loss_fn_attr_name, loss_fn)

            self.vertex_loss_fns.append(getattr(self, loss_fn_attr_name))

        # add each as a module so it gets moved to the right device.        
        self.vertex_loss_ws = vertex_loss_ws

        if len(self.vertex_loss_fns) != len(self.vertex_loss_ws):
            raise ValueError("vertex loss mismatch")

        self.position_embedding = build_position_encoding(self.model_dimension, kind="sine")
        self.level_embed = nn.Embedding(4, self.model_dimension)
        self.register_buffer("point_embedding", point_encoding(self.model_dimension * 2, max_len=self.number_control_points))

        if self.use_cls_token:
            self.cls_token = nn.Embedding(self.num_classes, self.model_dimension * 2)

        self.xy_embed = MLP(self.model_dimension, self.model_dimension, 2 if self.cls_agnostic else 2 * self.num_classes, 3)

        if self.ref_init == "random":
            self.reference_points = nn.Linear(self.model_dimension, 2)
        else:
            nn.init.constant_(self.xy_embed.layers[-1].bias.data, 0.0)
            nn.init.constant_(self.xy_embed.layers[-1].weight.data, 0.0)

        if self.model_dimension != 256:
            self.feature_proj = nn.ModuleList([
                nn.Linear(256, self.model_dimension),
                nn.Linear(256, self.model_dimension),
                nn.Linear(256, self.model_dimension),
                nn.Linear(256, self.model_dimension),                        
            ])
        else:
            self.feature_proj = None
                    
        activation = "relu"
        dec_n_points = 4
        nhead = 8
        
        self.feedforward_dimension = 1024
        decoder_layer = DeformableTransformerControlLayer(
            self.model_dimension, self.feedforward_dimension, self.dropout, activation, self.num_feature_levels, nhead, dec_n_points)

        if self.is_upsampling:
            decoder_layer = UpsamplingDecoderLayer(self.model_dimension, self.number_control_points, decoder_layer)
            self.start_idxs = decoder_layer.idxs[0]
            number_layers = decoder_layer.number_iterations # so we can get a final "layer".
            print(number_layers)
        else:
            number_layers = self.num_layers
            
        self.decoder = DeformableTransformerDecoder(
            decoder_layer, number_layers, return_intermediate=True, predict_in_box_space=self.predict_in_box_space)

        num_pred = self.decoder.num_layers            

        if self.iterative_refinement:
            self.xy_embed = _get_clones(self.xy_embed, num_pred)
            nn.init.constant_(self.xy_embed[0].layers[-1].bias.data, 0.0)
            self.decoder.xy_embed = self.xy_embed            
        else:
            self.xy_embed = nn.ModuleList([self.xy_embed for _ in range(num_pred)])

        self._reset_parameters()

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if ("xy_embed" in name):
                continue
            
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        if self.ref_init == "random":
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        
        normal_(self.level_embed.weight.data)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {
            "in_features": cfg.MODEL.BOUNDARY_HEAD.IN_FEATURES,
            "ref_init": cfg.MODEL.BOUNDARY_HEAD.POLY_INIT,
            "model_dim": cfg.MODEL.BOUNDARY_HEAD.MODEL_DIM,
            "number_layers": cfg.MODEL.BOUNDARY_HEAD.NUM_DEC_LAYERS,
            "number_control_points": cfg.MODEL.BOUNDARY_HEAD.POLY_NUM_PTS,
            "vis_period": cfg.VIS_PERIOD,
            "vertex_loss_fns": build_poly_losses(cfg, input_shape),
            "vertex_loss_ws": cfg.MODEL.BOUNDARY_HEAD.POLY_LOSS.WS,
            "is_upsampling": cfg.MODEL.BOUNDARY_HEAD.UPSAMPLING,
            "iterative_refinement": cfg.MODEL.BOUNDARY_HEAD.ITER_REFINE,
            "use_cls_token": cfg.MODEL.BOUNDARY_HEAD.USE_CLS_TOKEN,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic": cfg.MODEL.BOUNDARY_HEAD.CLS_AGNOSTIC_MASK,
            "predict_in_box_space": cfg.MODEL.BOUNDARY_HEAD.PRED_WITHIN_BOX,
            "prepool": cfg.MODEL.BOUNDARY_HEAD.PREPOOL,
            "dropout": cfg.MODEL.BOUNDARY_HEAD.DROPOUT,
            "deep_supervision": cfg.MODEL.BOUNDARY_HEAD.DEEP_SUPERVISION, 
        }
        
        ret.update(
            input_shape=input_shape,
        )
        
        return ret

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio    

    def forward(self, x, instances: List[Instances]):
        x = [x[f] for f in self.in_features]
        device = x[0].device            

        if self.prepool:
            if False:
                input_shapes = [x_.shape[-2:] for x_ in x]
                input_ys = [torch.linspace(-1, 1, s[0], device=device) for s in input_shapes]
                input_xs = [torch.linspace(-1, 1, s[1], device=device) for s in input_shapes]
                input_grid = [torch.stack(torch.meshgrid(y_, x_), dim=-1).unsqueeze(0).repeat(x[0].shape[0], 1, 1, 1) for y_, x_ in zip(input_ys, input_xs)]
                x = [F.grid_sample(x_, grid_) for x_, grid_ in zip(x, input_grid)]
            else:
                # todo, find out how the core reason this works so well.
                aligner = MultiROIPooler(
                    list(itertools.chain.from_iterable([[tuple(x_.shape[-2:])] for x_ in x])),
                    scales=(0.25, 0.125, 0.0625, 0.03125),
                    sampling_ratio=0,
                    pooler_type="ROIAlignV2",
                    assign_to_single_level=False)

                x = aligner(x, [Boxes(torch.Tensor([[0, 0, inst.image_size[1], inst.image_size[0]]]).to(x[0].device)) for inst in instances])        
            
        if self.feature_proj is not None:
            x = [self.feature_proj[i](x_.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) for i, x_ in enumerate(x)]

        number_levels = len(x)
        batch_size, feat_dim = x[0].shape[:2]
        
        if not self.training:
            no_instances = len(instances[0]) == 0
            if no_instances:
                instances[0].pred_masks = torch.zeros((0, 1, 4, 4), device=device)
                return instances

        masks = []
        pos_embeds = []
        srcs = []

        for l in range(number_levels):
            srcs.append(x[l])

            mask = torch.zeros((batch_size, x[l].shape[-2], x[l].shape[-1]), dtype=torch.bool, device=device)
            masks.append(mask)

            # todo, for non-pooled situation.. actually get the mask.
            f = NestedTensor(x[l], mask)
            pos_embeds.append(self.position_embedding(f))

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):            
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed.weight[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
            
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        if self.is_upsampling:
            # make sure to pull this out correctly.
            query_embed, tgt = torch.split(self.point_embedding[self.start_idxs], self.model_dimension, dim=1)
        else:
            query_embed, tgt = torch.split(self.point_embedding, self.model_dimension, dim=1)

        number_instances = [len(inst) for inst in instances]            
        max_instances = max(number_instances)
        box_preds_xyxy = pad_sequence([(inst.proposal_boxes.tensor if self.training else inst.pred_boxes.tensor) / torch.Tensor(2 * inst.image_size[::-1]).to(device)
                                       for inst in instances], batch_first=True)
            
        # normalize boxes            
        box_preds = [box_xyxy_to_cxcywh(
            (inst.proposal_boxes.tensor if self.training else inst.pred_boxes.tensor) / torch.Tensor(2 * inst.image_size[::-1]).to(device)) for inst in instances]
        box_preds = pad_sequence(box_preds, batch_first=True)
            
        query_embed = query_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, max_instances, -1, -1)
        tgt = tgt.unsqueeze(0).unsqueeze(0).expand(batch_size, max_instances, -1, -1)
        cls_token = None
 
        # sample with respect to box.
        if self.ref_init == "ellipse":
            if self.predict_in_box_space:
                reference_points = sample_ellipse_fast(
                    0.5 * torch.ones((batch_size, max_instances), device=device),
                    0.5 * torch.ones((batch_size, max_instances), device=device),
                    0.49 * torch.ones((batch_size, max_instances), device=device),
                    0.49 * torch.ones((batch_size, max_instances), device=device),
                    count=len(self.start_idxs) if self.is_upsampling else self.number_control_points)
            else:
                raise ValueError("todo")
        else:
            raise ValueError("todo")

        memory = src_flatten + lvl_pos_embed_flatten
        hs, inter_references = self.decoder(
            tgt, reference_points, memory, spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten, cls_token=cls_token, reference_boxes=box_preds_xyxy)
        
        outputs_coords = []
        for lvl in range(len(hs)):
            xy = self.xy_embed[lvl](hs[lvl])
            
            if self.predict_in_box_space:
                outputs_coord = (inverse_sigmoid(inter_references[lvl]) + xy).sigmoid()
            else:
                raise ValueError("todo")

            # remove any padded entries.            
            outputs_coords.append(outputs_coord)

        # unpad and flatten across batch dim.
        for i in range(len(outputs_coords)):
            output_coords = outputs_coords[i]
            outputs_coords[i] = torch.cat([output_coords[j, :number_instances[j]] for j in range(batch_size)])

        if self.training:
            vertex_losses = {}        

            for vertex_loss_fn in self.vertex_loss_fns:
                output_losses = []

                if not self.deep_supervision:
                    outputs_coords = [outputs_coords[-1]]
                
                for lid, output_coords in enumerate(outputs_coords):
                    output_loss = vertex_loss_fn(output_coords, instances, lid=lid)
                    output_losses.append(output_loss)

                vertex_losses[vertex_loss_fn.name] = torch.stack(output_losses)

        
            ret_loss = {
                "loss_{0}".format(vertex_loss_fn.name): vertex_loss_w * vertex_losses[vertex_loss_fn.name].mean()
                for vertex_loss_w, vertex_loss_fn in zip(self.vertex_loss_ws, self.vertex_loss_fns)
            }            

            # hack to monitor this.
            if hasattr(self.vertex_loss_fns[0], "inv_smoothness"):
                ret_loss["loss_smooth"] = torch.Tensor([self.vertex_loss_fns[0].inv_smoothness]).to(device)

            return ret_loss

        num_boxes_per_image = [len(i) for i in instances]

        # always take the last layer's outputs for now.
        pred_polys_per_image = outputs_coords[-1].split(number_instances, dim=0)
        for pred_polys, instance in zip(pred_polys_per_image, instances):
            instance.pred_polys = pred_polys
        
        return instances
        
def build_poly_losses(cfg, input_shape):
    """
    Build polygon losses `cfg.MODEL.BOUNDARY_HEAD.POLY_LOSS.NAMES`.
    """

    losses = []
    for name in cfg.MODEL.BOUNDARY_HEAD.POLY_LOSS.NAMES:
        losses.append(POLY_LOSS_REGISTRY.get(name)(cfg, input_shape))

    return losses
