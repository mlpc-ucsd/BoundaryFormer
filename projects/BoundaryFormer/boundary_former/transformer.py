import math
import torch
from torch import nn
from torch.nn import functional as F

from boundary_former.layers.deform_attn import MSDeformAttn
from boundary_former.utils import _get_clones, inverse_sigmoid

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def point_encoding(d_model, max_len=64):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)

    return pe[:, 0, :]

# Non-pooled case e.g. N x Q
class DeformableTransformerControlLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=4, n_points=4):
        super().__init__()

        self.d_model = d_model
        
        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.emulate_pooling = True

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None, lid=None, cls_token=None, reference_boxes=None):
        batch_size, num_query, num_control, dim_control = tgt.shape
        num_lvl = reference_points.shape[-2]

        tgt = tgt.view(-1, num_control, dim_control)
        query_pos = query_pos.view(-1, num_control, dim_control)        
                
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        tgt = tgt.reshape(batch_size, num_query * num_control, dim_control)
        query_pos = query_pos.reshape(batch_size, num_query * num_control, dim_control)

        if self.emulate_pooling:
            # pass these through directly so we can compute a box.
            pass
        else:
            reference_points = reference_points.view(batch_size, -1, num_lvl, 2)

        # for cross attn, reshape in the query dimension
        # cross attention. Note: while reference points might be fixed, sampling offsets are a function of
        # with_pos_embed(tgt, query_pos)
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask, reference_boxes=reference_boxes)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # ffn
        tgt = self.forward_ffn(tgt)
        return tgt.view(batch_size, num_query, num_control, dim_control), None

class UpsamplingDecoderLayer(nn.Module):    
    def __init__(self, model_dimension, max_control_points, inner):
        super().__init__()

        self.model_dimension = model_dimension
        self.base_control_points = 8
        self.max_control_points = max_control_points        
        self.inner = inner

        self.register_buffer("point_embedding", point_encoding(self.model_dimension * 2, max_len=self.max_control_points))
        self.number_iterations = int(math.log2(self.max_control_points // self.base_control_points)) + 1
        print("{0} to {1} over {2} layers".format(self.base_control_points, self.max_control_points, self.number_iterations))

        self.idxs = []
        for iter_idx in range(self.number_iterations):
            if iter_idx == 0:
                idxs = torch.arange(0, self.max_control_points + 1, self.max_control_points // self.base_control_points)
                self.idxs.append(idxs[:-1])                
            else:
                new_idxs = ((idxs[:-1] + idxs[1:]) / 2).long()
                self.idxs.append(new_idxs)

                # add back the end?
                idxs = torch.sort(torch.cat((idxs, new_idxs)))[0]

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None, lid=0, cls_token=None, reference_boxes=None):
        # presumably, x is an N x K where K is some number of points.
        # we insert the position encoding for K more points,
        # at the same time, we interpolate the "reference" points to be:
        #  1. possibly an interpolation of the previous reference points?
        #  2. possibly an interpolation of the previous _predictions_.
        # assume REFINE_ITER is on.
        # add the current reference_points encoding to query_pos?
        # this might be a no-op, but just make sure query_pos is correct.
        if lid == 0:
            return self.inner(tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask, cls_token=cls_token, reference_boxes=reference_boxes)

        # let the last one also adjust?
        # if lid == self.number_iterations:
        #     return self.inner(tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask)

        orig_batch_size, num_query, num_control, dim_control = tgt.shape
        num_lvl = reference_points.shape[-2]
            
        tgt = tgt.view(-1, num_control, dim_control)
        query_pos = query_pos.view(-1, num_control, dim_control)
        reference_points = reference_points.view(-1, num_control, num_lvl, 2)

        batch_size = len(tgt)
            
        iteration_idx = lid
        insert_query_pos, insert_tgt = torch.split(self.point_embedding[self.idxs[iteration_idx]], self.model_dimension, dim=1)

        # assume ITER_REFINE is on. interpolate between the _predicted_ xy.
        # concatenate adjacent feature representations, predict some _t_ indicating where to put this point.
        insert_reference_points = (reference_points + torch.roll(reference_points, -1, dims=1)) / 2.0

        insert_query_pos = insert_query_pos.unsqueeze(0).expand(batch_size, -1, -1)
        insert_tgt = insert_tgt.unsqueeze(0).expand(batch_size, -1, -1)

        # insert at every other and continue.
        query_pos = torch.stack((query_pos, insert_query_pos), dim=2).view(batch_size, -1, self.model_dimension)
        tgt = torch.stack((tgt, insert_tgt), dim=2).view(batch_size, -1, self.model_dimension)

        reference_points = torch.stack((reference_points, insert_reference_points), dim=2).view(batch_size, -1, reference_points.shape[-2], 2)

        tgt = tgt.view(orig_batch_size, num_query, -1, dim_control)
        query_pos = query_pos.view(orig_batch_size, num_query, -1, dim_control)
        reference_points = reference_points.view(orig_batch_size, num_query, -1, num_lvl, 2)
            
        insert_reference_points = insert_reference_points.view(orig_batch_size, num_query, -1, num_lvl, 2)
            
        # needs to return what it changed.
        return self.inner(tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask, cls_token=cls_token, reference_boxes=reference_boxes)[0], (tgt, query_pos, insert_reference_points)
    
class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, predict_in_box_space=False):
        super().__init__()

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.predict_in_box_space = predict_in_box_space

        # iterative refinement.
        self.xy_embed = None        

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, cls_token=None, reference_boxes=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if self.return_intermediate:
                intermediate_reference_points.append(reference_points)
            
            if reference_points.shape[-1] == 4:
                # not supported
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, :, None] * src_valid_ratios[:, None, None]                                        

            output, inserted = layer(
                output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask, lid, cls_token=cls_token, reference_boxes=reference_boxes)
            
            if not (cls_token is None):
                cls_token_tgt = output[:, -1]

                # reunite it with the pos.
                cls_token = torch.cat((cls_token[:, :cls_token_tgt.shape[-1]], cls_token_tgt), dim=-1)                
                output = output[:, :-1]

            if not (inserted is None):
                # note, we probably want the xy for the _first_ time a query appeared.
                tgt, query_pos, inserted_reference_points = inserted
                batch_size = len(tgt)

                max_instances = tgt.shape[1]
                reference_points = torch.stack((reference_points, inserted_reference_points[:, :, :, 0]), dim=3).view(batch_size, max_instances, -1, 2)
                intermediate_reference_points[-1] = reference_points 

            if self.xy_embed is not None:
                tmp = self.xy_embed[lid](output)
                    
                assert reference_points.shape[-1] == 2
                new_reference_points = tmp + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
                # unclear if detach() matters.
                reference_points = new_reference_points #.detach()                

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return intermediate, intermediate_reference_points

        return output, reference_points    
