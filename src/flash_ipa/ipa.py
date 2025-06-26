# Copyright 2025 Anonymous
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited

import torch
import torch.nn as nn
from typing import Optional, List, Sequence
import math
import torch.nn.functional as F
from einops import rearrange
from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_varlen_func
from flash_ipa.linear import Linear
from flash_ipa.rigid import Rigid
from dataclasses import dataclass

attn_dtype_dict = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


@dataclass
class IPAConfig:
    use_flash_attn: bool = True
    attn_dtype: str = "bf16"  # "fp16", "bf16", "fp32". For flash ipa, bf16 or fp16. For original, fp32.
    use_packed: bool = True
    c_s: int = 256
    c_z: int = 128
    c_hidden: int = 128
    no_heads: int = 8
    z_factor_rank: int = 2  # 0 for no factorization
    no_qk_points: int = 8
    no_v_points: int = 12
    seq_tfmr_num_heads: int = 4
    seq_tfmr_num_layers: int = 2
    num_blocks: int = 6


class InvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22, with flash IPA.
    """

    def __init__(
        self,
        ipa_conf: IPAConfig,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(InvariantPointAttention, self).__init__()
        self._ipa_conf = ipa_conf

        self.use_flash_attn = ipa_conf.use_flash_attn
        self.attn_dtype = attn_dtype_dict[ipa_conf.attn_dtype]
        self.use_packed = ipa_conf.use_packed

        self.c_s = ipa_conf.c_s
        self.c_z = ipa_conf.c_z
        self.c_hidden = ipa_conf.c_hidden
        self.no_heads = ipa_conf.no_heads
        self.no_qk_points = ipa_conf.no_qk_points
        self.no_v_points = ipa_conf.no_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)
        if self.c_z > 0:
            self.linear_b = Linear(self.c_z, self.no_heads)
            self.down_z = Linear(self.c_z, self.c_z // 4)

        self.head_weights = nn.Parameter(torch.zeros((ipa_conf.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

        self.headdim_eff = max(
            self._ipa_conf.c_hidden + 5 * self.no_qk_points + (self._ipa_conf.z_factor_rank * self.no_heads),
            self._ipa_conf.c_hidden + 3 * self.no_v_points + (self._ipa_conf.z_factor_rank * self.c_z // 4),
        )

        if self.headdim_eff > 256:
            assert (
                self.use_flash_attn is False or self.attn_dtype == torch.float16
            ), "For headdim_eff > 256, you must use either naive attention or FFPA, which requires fp16 dtype."

    def flash_ipa_fwd(self, q, k, v, q_pts, k_pts, v_pts, z_factor_1, z_factor_2, r, mask):
        """
        Compute squared norm components (used for SE(3) invariance part)
        """
        q_pts_norm_sq = torch.norm(q_pts, dim=-1) ** 2
        k_pts_norm_sq = torch.norm(k_pts, dim=-1) ** 2

        """
        Compute non-zero padding (used for SE(3) invariance part)
        """
        head_weights = self.softplus(self.head_weights)
        head_weights = head_weights * math.sqrt(1.0 / (3 * (self.no_qk_points * 9.0 / 2)))
        q_pad = torch.ones_like(q_pts_norm_sq)
        k_pad = torch.ones_like(k_pts_norm_sq) * (-0.5) * head_weights.view(1, 1, -1, 1)

        """
        Compute pair bias factors
        """
        if z_factor_1 is not None and z_factor_2 is not None:
            # z_factor_1 has shape [B, N_res, rank, C_z]
            z_comb = torch.cat([z_factor_1.unsqueeze(1), z_factor_2.unsqueeze(1)], dim=1)
            b = self.linear_b(z_comb)
            b1 = b[:, 0, :, :, :].permute(0, 1, 3, 2)  # B, N_res, H, rank
            b2 = b[:, 1, :, :, :].permute(0, 1, 3, 2)  # B, N_res, H, rank

            z_comb_down = self.down_z(z_comb)
            z_factor_1 = z_comb_down[:, 0, :, :, :]  # B, N_res, rank, C_z//4
            z_factor_2 = z_comb_down[:, 1, :, :, :]  # B, N_res, rank, C_z//4

        """
        Compute q_aggregated
        """
        if z_factor_1 is not None:
            q_aggregated = torch.cat(
                [q, q_pts.view(q_pts.shape[0], q_pts.shape[1], q_pts.shape[2], -1), q_pts_norm_sq, q_pad, b1], dim=-1
            )
        else:
            q_aggregated = torch.cat(
                [q, q_pts.view(q_pts.shape[0], q_pts.shape[1], q_pts.shape[2], -1), q_pts_norm_sq, q_pad], dim=-1
            )
        """
        Compute k_aggregated
        """
        k_scaled = k * math.sqrt(1.0 / (3 * self.c_hidden))
        k_pts_scaled = k_pts.view(k_pts.shape[0], k_pts.shape[1], k_pts.shape[2], -1) * head_weights.view(1, 1, -1, 1)
        k_pts_norm_sq_scaled = k_pts_norm_sq * (-0.5) * head_weights.view(1, 1, -1, 1)
        if z_factor_2 is not None:
            k_aggregated = torch.cat([k_scaled, k_pts_scaled, k_pad, k_pts_norm_sq_scaled, b2], dim=-1)
        else:
            k_aggregated = torch.cat([k_scaled, k_pts_scaled, k_pad, k_pts_norm_sq_scaled], dim=-1)

        """
        Compute v_aggregated
        """
        if z_factor_2 is not None:
            v_aggregated = torch.cat(
                [
                    v,
                    v_pts.view(*v_pts.shape[:3], -1),
                    z_factor_2.view(*z_factor_2.shape[:2], 1, -1).expand(-1, -1, self.no_heads, -1),
                ],
                dim=-1,
            )
        else:
            v_aggregated = torch.cat([v, v_pts.view(v_pts.shape[0], v_pts.shape[1], v_pts.shape[2], -1)], dim=-1)

        if mask is None:
            mask = torch.ones((q.shape[0], q.shape[1]), device=q.device, dtype=torch.bool)

        """
        Pass through FA2 or FFPA depending on headdim size
        """

        if self.headdim_eff <= 256:  # use FA2
            # FA2 requires that QKV have same size for last dimension. So just choose the smallest possible size.
            max_dim_sz = max(q_aggregated.shape[-1], k_aggregated.shape[-1], v_aggregated.shape[-1])
            q_aggregated = F.pad(q_aggregated, (0, max_dim_sz - q_aggregated.shape[-1]), value=0.0)
            k_aggregated = F.pad(k_aggregated, (0, max_dim_sz - k_aggregated.shape[-1]), value=0.0)
            v_aggregated = F.pad(v_aggregated, (0, max_dim_sz - v_aggregated.shape[-1]), value=0.0)
            if self.use_packed:
                qkv = torch.cat([q_aggregated.unsqueeze(2), k_aggregated.unsqueeze(2), v_aggregated.unsqueeze(2)], dim=2)
                (
                    qkv,
                    indices,
                    cu_seqlens,
                    max_seqlen,
                    _,
                ) = unpad_input(qkv, mask)

                if qkv.dtype != self.attn_dtype:
                    qkv = qkv.to(self.attn_dtype)

                attn_res = flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, softmax_scale=1)

            else:
                q_aggregated, indices, cu_seqlens_q, max_seqlen_q, _ = unpad_input(q_aggregated, mask)
                k_aggregated, _, cu_seqlens_k, max_seqlen_k, _ = unpad_input(k_aggregated, mask)
                v_aggregated, _, cu_seqlens_v, max_seqlen_v, _ = unpad_input(v_aggregated, mask)

                if (
                    q_aggregated.dtype != self.attn_dtype
                    or k_aggregated.dtype != self.attn_dtype
                    or v_aggregated.dtype != self.attn_dtype
                ):
                    q_aggregated = q_aggregated.to(self.attn_dtype)
                    k_aggregated = k_aggregated.to(self.attn_dtype)
                    v_aggregated = v_aggregated.to(self.attn_dtype)

                attn_res = flash_attn_varlen_func(
                    q_aggregated,
                    k_aggregated,
                    v_aggregated,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    softmax_scale=1,
                )
            attn_res = pad_input(
                attn_res,
                indices=indices,
                batch=q.shape[0],
                seqlen=q.shape[1],
            )
        else:
            raise ValueError(f"self.headdim_eff has to be <= 256 for FA2 to work: {self.headdim_eff}")

        if attn_res.dtype != torch.float32:
            attn_res = attn_res.float()

        if z_factor_2 is not None:
            attn_res = attn_res[
                :, :, :, : self.c_hidden + 3 * self.no_v_points + self._ipa_conf.z_factor_rank * z_factor_2.shape[-1]
            ]
        else:
            attn_res = attn_res[:, :, :, : self.c_hidden + 3 * self.no_v_points]

        o = attn_res[:, :, :, : self.c_hidden]
        o = flatten_final_dims(o, 2)

        # B,L,H,D
        o_pt = attn_res[:, :, :, self.c_hidden : self.c_hidden + 3 * self.no_v_points]
        # [*, H, 3, N_res, P_v]
        o_pt = rearrange(o_pt, "B L H (P_v r) -> B H r L P_v", P_v=self.no_v_points)

        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_dists = torch.sqrt(torch.sum(o_pt**2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(o_pt_dists, 2)

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        # calculate o_pair
        if z_factor_1 is not None and z_factor_2 is not None:
            o_pair = attn_res[:, :, :, self.c_hidden + 3 * self.no_v_points :].view(
                *attn_res.shape[:3], self._ipa_conf.z_factor_rank, -1
            )  # B, L, H, rank, C_z//4
            o_pair = torch.einsum("b n r d, b n h r d -> b n h d", z_factor_1, o_pair)
            o_pair = flatten_final_dims(o_pair, 2)
            o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]
        else:
            o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats]

        s = self.linear_out(torch.cat(o_feats, dim=-1))

        return s

    def slow_ipa_fwd(self, q, k, v, q_pts, k_pts, v_pts, z, r, mask, _offload_inference=False):
        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]

        if not z is None:
            b = self.linear_b(z[0])

            if _offload_inference:
                z[0] = z[0].cpu()

        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))

        if not z is None:
            a += math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1))

        # [*, N_res, N_res, H, P_q, 3]
        pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_displacement**2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(*((1,) * len(pt_att.shape[:-2]) + (-1, 1)))
        head_weights = head_weights * math.sqrt(1.0 / (3 * (self.no_qk_points * 9.0 / 2)))
        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        a = a + pt_att
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(a, v.transpose(-2, -3)).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v]
        o_pt = torch.sum(
            (a[..., None, :, :, None] * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]),
            dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_dists = torch.sqrt(torch.sum(o_pt**2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(o_pt_dists, 2)

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if not z is None:
            if _offload_inference:
                z[0] = z[0].to(o_pt.device)

            # [*, N_res, H, C_z // 4]
            pair_z = self.down_z(z[0])
            o_pair = torch.matmul(a.transpose(-2, -3), pair_z)

            # [*, N_res, H * C_z // 4]
            o_pair = flatten_final_dims(o_pair, 2)

            o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]
        else:
            o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats]

        # [*, N_res, C_s]
        s = self.linear_out(torch.cat(o_feats, dim=-1))
        return s

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        z_factor_1: Optional[torch.Tensor],
        z_factor_2: Optional[torch.Tensor],
        r: Rigid,
        mask: torch.Tensor,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        if not z is None:
            if _offload_inference:
                z = _z_reference_list
            else:
                z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3))

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(kv_pts, [self.no_qk_points, self.no_v_points], dim=-2)

        if self.use_flash_attn:
            s = self.flash_ipa_fwd(
                q,
                k,
                v,
                q_pts,
                k_pts,
                v_pts,
                z_factor_1,
                z_factor_2,
                r,
                mask=mask,
            )

        else:
            s = self.slow_ipa_fwd(
                q,
                k,
                v,
                q_pts,
                k_pts,
                v_pts,
                z,
                r,
                mask=mask,
                _offload_inference=_offload_inference,
            )
        return s


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


"""
Unpadding and padding operations for FlashAttention
"""


def unpad_input(hidden_states, attention_mask, unused_mask=None):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
        unused_mask: (batch, seqlen), bool / int, 1 means the element is allocated but unused.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask + unused_mask.
        indices: (total_nnz), the indices of masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
        seqused: (batch), returns the number of tokens selected in attention_mask + unused_mask.
    """
    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        rearrange(hidden_states, "b s ... -> (b s) ...")[indices],
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )


def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    dim = hidden_states.shape[1:]
    output = torch.zeros((batch * seqlen), *dim, device=hidden_states.device, dtype=hidden_states.dtype)
    output[indices] = hidden_states
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


class StructureModuleTransition(nn.Module):
    def __init__(self, c):
        super(StructureModuleTransition, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(self.c)

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        s = s + s_initial
        s = self.ln(s)

        return s


class EdgeTransition(nn.Module):
    def __init__(
        self,
        *,
        mode,
        node_embed_size,
        edge_embed_in,
        edge_embed_out,
        z_factor_rank=0,
        num_layers=2,
        node_dilation=2,
    ):
        super(EdgeTransition, self).__init__()

        self.mode = mode
        self.z_factor_rank = z_factor_rank
        assert mode in ["1d", "2d"], f"Invalid mode: {mode}. Must be '1d' or '2d'."
        bias_embed_size = node_embed_size // node_dilation

        self.initial_embed = Linear(node_embed_size, bias_embed_size, init="relu")
        if mode == "1d":
            self.edge_bias_linear = Linear(bias_embed_size, 4 * self.z_factor_rank * bias_embed_size, init="final")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = Linear(hidden_size, edge_embed_out, init="final")
        self.layer_norm = nn.LayerNorm(edge_embed_out)

    def forward(self, node_embed, edge_embed, z_factor_1=None, z_factor_2=None):
        if edge_embed is not None:
            return self.fwd_2d(node_embed, edge_embed)
        elif z_factor_1 is not None and z_factor_2 is not None:
            return self.fwd_1d(node_embed, z_factor_1, z_factor_2)

    def fwd_1d(self, node_embed, z_factor_1, z_factor_2):
        node_embed = self.initial_embed(node_embed)  # B,L,D

        batch_size, num_res, _ = node_embed.shape
        rank = z_factor_1.shape[2]

        edge_bias = self.edge_bias_linear(node_embed)
        edge_bias = rearrange(edge_bias, "b l (n r d) -> b n l r d", r=self.z_factor_rank, n=2)  # B,2,L,R,2*D

        z_agg = torch.cat(
            [z_factor_1[:, None, :, :, :], z_factor_2[:, None, :, :, :]],
            axis=1,
        )

        edge_embed = torch.cat([z_agg, edge_bias], axis=-1) / math.sqrt(2)

        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        z_factor_1 = edge_embed[:, 0, :, :, :]
        z_factor_2 = edge_embed[:, 1, :, :, :]
        return z_factor_1, z_factor_2

    def fwd_2d(self, node_embed, edge_embed):
        node_embed = self.initial_embed(node_embed)
        batch_size, num_res, _ = node_embed.shape
        edge_bias = torch.cat(
            [
                torch.tile(node_embed[:, :, None, :], (1, 1, num_res, 1)),
                torch.tile(node_embed[:, None, :, :], (1, num_res, 1, 1)),
            ],
            axis=-1,
        )
        edge_embed = torch.cat([edge_embed, edge_bias], axis=-1).reshape(batch_size * num_res**2, -1)  # B*L*L,D
        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        edge_embed = edge_embed.reshape(batch_size, num_res, num_res, -1)
        return edge_embed


class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s, use_rot_updates):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super(BackboneUpdate, self).__init__()

        self.c_s = c_s
        self._use_rot_updates = use_rot_updates
        update_dim = 6 if use_rot_updates else 3
        self.linear = Linear(self.c_s, update_dim, init="final")

    def forward(self, s: torch.Tensor):
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector
        """
        # [*, 6]
        update = self.linear(s)

        return update
