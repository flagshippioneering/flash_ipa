"""
Code adapted from
https://github.com/microsoft/protein-frame-flow/blob/main/models/utils.py
"""

import math
import torch

NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE


def calc_distogram(pos, min_bin, max_bin, num_bins):
    dists_2d = torch.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[..., None]
    lower = torch.linspace(min_bin, max_bin, num_bins, device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram


def knn_indices(pos, k):

    # pos: [B, L, 3]
    B, L, _ = pos.shape
    x = pos.unsqueeze(2)  # [B, L, 1, 3]
    y = pos.unsqueeze(1)  # [B, 1, L, 3]
    dist = torch.norm(x - y, dim=-1)  # [B, L, L]

    # Prevent self-matching
    dist += torch.eye(L, device=pos.device).unsqueeze(0) * 1e6
    dist_k, idx_k = torch.topk(dist, k=k, dim=-1, largest=False)  # [B, L, k]
    return dist_k, idx_k


def calc_distogram_knn(pos, k, min_bin, max_bin, num_bins):
    dist_k, idx_k = knn_indices(pos, k)  # [B, L, k]
    dists = dist_k.unsqueeze(-1)

    # Bin edges
    lower = torch.linspace(min_bin, max_bin, num_bins, device=pos.device)  # [num_bins]
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=0)  # [num_bins]

    # Create one-hot bin assignment
    dgram = ((dists > lower) & (dists < upper)).type(pos.dtype)  # [B, L, k, num_bins]

    return dgram, idx_k


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size // 2, device=indices.device)
    pos_embedding_sin = torch.sin(indices[..., None] * math.pi / (max_len ** (2 * K[None] / embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(indices[..., None] * math.pi / (max_len ** (2 * K[None] / embed_size))).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def check_config_ipa(ipa_conf, mode: str):
    if mode == "orig_no_bias":
        assert (
            ipa_conf.c_z == 0 and ipa_conf.z_factor_rank == 0 and ipa_conf.use_flash_attn == False
        ), "Expecting ipa_conf.c_z == 0 and ipa_conf.z_factor_rank == 0 and ipa_conf.use_flash_attn == False, but got {ipa_conf.c_z}, {ipa_conf.z_factor_rank}, {ipa_conf.use_flash_attn}."
    elif mode == "orig_2d_bias":
        assert (
            ipa_conf.c_z > 0 and ipa_conf.z_factor_rank == 0 and ipa_conf.use_flash_attn == False
        ), "Expecting ipa_conf.c_z > 0 and ipa_conf.z_factor_rank == 0 and ipa_conf.use_flash_attn == False, but got {ipa_conf.c_z}, {ipa_conf.z_factor_rank}, {ipa_conf.use_flash_attn}."
    elif mode == "flash_no_bias":
        assert (
            ipa_conf.c_z == 0 and ipa_conf.z_factor_rank == 0 and ipa_conf.use_flash_attn == True
        ), "Expecting ipa_conf.c_z == 0 and ipa_conf.z_factor_rank == 0 and ipa_conf.use_flash_attn == True, but got {ipa_conf.c_z}, {ipa_conf.z_factor_rank}, {ipa_conf.use_flash_attn}."
    elif mode == "flash_1d_bias":
        assert (
            ipa_conf.c_z > 0 and ipa_conf.z_factor_rank > 0 and ipa_conf.use_flash_attn == True
        ), "Expecting ipa_conf.c_z > 0 and ipa_conf.z_factor_rank > 0 and ipa_conf.use_flash_attn == True, but got {ipa_conf.c_z}, {ipa_conf.z_factor_rank}, {ipa_conf.use_flash_attn}."
    elif mode == "flash_2d_factorize_bias":
        assert (
            ipa_conf.c_z > 0 and ipa_conf.z_factor_rank > 0 and ipa_conf.use_flash_attn == True
        ), "Expecting ipa_conf.c_z > 0 and ipa_conf.z_factor_rank > 0 and ipa_conf.use_flash_attn == True, but got {ipa_conf.c_z}, {ipa_conf.z_factor_rank}, {ipa_conf.use_flash_attn}."
    else:
        raise ValueError(
            f"Invalid mode: {mode}. Must be one of ['orig_no_bias', 'orig_2d_bias', 'flash_no_bias', 'flash_1d_bias', 'flash_2d_factorize_bias']."
        )
