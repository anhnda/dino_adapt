# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch
from torch import nn, Tensor

import math
import torch.nn.functional as F
from torch import nn
import torch


def get_k_tensor_constrained(ar, sub_seq, offset=40, lag=40, alpha=1, beta=0.4, gamma=2):
    """
    Zero-warning ONNX version. Requires sub_seq to be a tensor for tracing.
    Call this version during ONNX export to eliminate all warnings.
    
    Important: Ensure sub_seq is already a tensor before calling this function!
    """
    batch_size, N = ar.shape
    device = ar.device
    # return torch.tensor([100] * batch_size, device = device)

    # Assume sub_seq is already a tensor (no conversion during tracing)
    # sub_seq = sub_seq.to(device)

    # Calculate per-sequence k ranges
    k_end_global = N - offset - lag
    k_end_per_seq = sub_seq - lag
    k_end_per_seq = torch.clamp(k_end_per_seq, max=k_end_global)

    # Use fixed maximum range to avoid dynamic operations
    max_possible_k = N - offset - lag
    k_range = torch.arange(offset, max_possible_k, device=device, dtype=torch.long)
    
    # Create masks without control flow
    k_range_expanded = k_range.unsqueeze(0)
    k_end_expanded = k_end_per_seq.unsqueeze(1)
    valid_k_mask = k_range_expanded < k_end_expanded

    num_k = k_range.shape[0]
    ar_expanded = ar.unsqueeze(1)
    k_expanded = k_range.unsqueeze(0).unsqueeze(2)

    # Create segment masks
    indices = torch.arange(N, device=device, dtype=torch.long).unsqueeze(0).unsqueeze(0)
    mask_A1 = indices < k_expanded
    mask_A2 = indices >= (k_expanded + lag)

    # Broadcast
    mask_A1 = mask_A1.expand(batch_size, num_k, N)
    mask_A2 = mask_A2.expand(batch_size, num_k, N)
    ar_broadcast = ar_expanded.expand(batch_size, num_k, N)

    # Calculations using only tensor operations
    A1_lengths = mask_A1.sum(dim=2, keepdim=True)
    x_coords = torch.arange(N, device=device, dtype=torch.float32).expand(batch_size, num_k, N)

    zeros_float = torch.zeros_like(ar_broadcast)
    A1_vals = torch.where(mask_A1, ar_broadcast, zeros_float)
    A1_x = torch.where(mask_A1, x_coords, zeros_float)

    k_start_vals = k_range.unsqueeze(0).expand(batch_size, num_k).float()
    A1_x_adjusted = A1_x - k_start_vals.unsqueeze(2) * mask_A1.float()
    A1_x_adjusted = torch.where(mask_A1, A1_x_adjusted, zeros_float)

    # Linear regression calculations
    n = A1_lengths.squeeze(2).float()
    sum_x = A1_x_adjusted.sum(dim=2)
    sum_y = A1_vals.sum(dim=2)
    sum_xy = (A1_x_adjusted * A1_vals).sum(dim=2)
    sum_x2 = (A1_x_adjusted ** 2).sum(dim=2)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = n * sum_x2 - sum_x ** 2

    # Use torch.full_like to avoid torch.tensor() warnings
    eps = torch.full_like(denominator, 1e-10)
    denominator = torch.where(denominator.abs() < eps, eps, denominator)
    slope_A1 = numerator / denominator

    # Variance calculations
    A1_counts = mask_A1.sum(dim=2).float()
    A2_counts = mask_A2.sum(dim=2).float()

    A1_mean = A1_vals.sum(dim=2) / torch.clamp(A1_counts, min=1)
    A1_vals_centered = A1_vals - A1_mean.unsqueeze(2) * mask_A1.float()
    A1_var = (A1_vals_centered ** 2 * mask_A1.float()).sum(dim=2) / torch.clamp(A1_counts, min=1)

    A2_vals = torch.where(mask_A2, ar_broadcast, zeros_float)
    A2_mean = A2_vals.sum(dim=2) / torch.clamp(A2_counts, min=1)
    A2_vals_centered = A2_vals - A2_mean.unsqueeze(2) * mask_A2.float()
    A2_var = (A2_vals_centered ** 2 * mask_A2.float()).sum(dim=2) / torch.clamp(A2_counts, min=1)

    # Final scoring and selection
    scores = alpha * slope_A1.abs() + beta * A1_var - gamma * A2_var
    neg_inf = torch.full_like(scores, -1e10)
    scores = torch.where(valid_k_mask, scores, neg_inf)

    best_indices = scores.argmax(dim=1)
    best_k = k_range[best_indices]

    result = best_k + lag
    result = torch.clamp(result, max=sub_seq)

    return result

def hash_v(v):
    thresholds = list(range(50, 301, 5))  # [15, 20, ..., 300]
    best = thresholds[0]  # default to minimum
    for t in thresholds:
        if v >= t:
            best = t
        else:
            break
    return best


def gen_encoder_output_proposals(memory:Tensor, memory_padding_mask:Tensor, spatial_shapes:Tensor, learnedwh=None):
    """
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
        - learnedwh: 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    N_, S_, C_ = memory.shape
    base_scale = 4.0
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

        grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                        torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1) # H_, W_, 2

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale

        if learnedwh is not None:
            wh = torch.ones_like(grid) * learnedwh.sigmoid() * (2.0 ** lvl)
        else:
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)

        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += (H_ * W_)

    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals)) # unsigmoid
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

    output_memory = memory
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

    return output_memory, output_proposals


class RandomBoxPerturber():
    def __init__(self, x_noise_scale=0.2, y_noise_scale=0.2, w_noise_scale=0.2, h_noise_scale=0.2) -> None:
        self.noise_scale = torch.Tensor([x_noise_scale, y_noise_scale, w_noise_scale, h_noise_scale])

    def __call__(self, refanchors: Tensor) -> Tensor:
        nq, bs, query_dim = refanchors.shape
        device = refanchors.device

        noise_raw = torch.rand_like(refanchors)
        noise_scale = self.noise_scale.to(device)[:query_dim]

        new_refanchors = refanchors * (1 + (noise_raw - 0.5) * noise_scale)
        return new_refanchors.clamp_(0, 1)


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


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


def _get_activation_fn(activation, d_model=256, batch_dim=0):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu

    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos