"""
Code adapted from
https://github.com/microsoft/protein-frame-flow/blob/main/models/edge_embedder.py
"""

import torch
from torch import nn
from flash_ipa import utils
from einops import rearrange
from typing import Optional
from dataclasses import dataclass
from flash_ipa.factorizer import LinearFactorizer


@dataclass
class EdgeEmbedderConfig:
    z_factor_rank: int = 2  # Rank of the factorization of the edge embedding
    single_bias_transition_n: int = 2  # Not used
    c_s: int = 256  # Size of the node embedding
    c_p: int = 128  # Size of the edge embedding
    relpos_k: int = 64  # Not used
    use_rbf: bool = True  # Not used
    num_rbf: int = 32  # Not used
    feat_dim: int = 64  # Hidden dimension of the edge embedder
    num_bins: int = 22  # Number of bins for the distogram
    self_condition: bool = True  # Not used
    mode: str = "flash_1d_bias"  # "orig_no_bias", "orig_2d_bias", "flash_no_bias", "flash_1d_bias", "flash_2d_factorize_bias"
    k: int = 10  # Number of nearest neighbors for the distogram
    max_len: Optional[int] = None  # Maximum length of the input sequence


class EdgeEmbedder(nn.Module):

    def __init__(self, module_cfg):
        super(EdgeEmbedder, self).__init__()
        self._cfg = module_cfg

        self.c_s = self._cfg.c_s
        self.c_p = self._cfg.c_p
        self.feat_dim = self._cfg.feat_dim
        self.mode = self._cfg.mode
        self.max_len = self._cfg.max_len

        total_edge_feats = self.feat_dim * 3 + self._cfg.num_bins * 2

        if self.mode == "flash_1d_bias":
            self.k = self._cfg.k
            self.z_factor_rank = self._cfg.z_factor_rank
            self.linear_s_p = nn.Linear(self.c_s, self.feat_dim * self.z_factor_rank * 2)
            self.linear_relpos = nn.Linear(self.feat_dim, self.feat_dim * self.z_factor_rank * 4)
            self.linear_t_distogram = nn.Linear(
                self.k * (self._cfg.num_bins + self.feat_dim), self._cfg.num_bins * self.z_factor_rank * 2
            )
            self.linear_sc_distogram = nn.Linear(
                self.k * (self._cfg.num_bins + self.feat_dim), self._cfg.num_bins * self.z_factor_rank * 2
            )
        elif self.mode in ["orig_2d_bias", "flash_2d_factorize_bias"]:
            self.linear_s_p = nn.Linear(self.c_s, self.feat_dim)
            self.linear_relpos = nn.Linear(self.feat_dim, self.feat_dim)
            if self.mode == "flash_2d_factorize_bias":
                assert self.max_len is not None, "max_len must be provided for flash_2d_factorize_bias mode"
                self.z_factor_rank = self._cfg.z_factor_rank
                self.factorizer = LinearFactorizer(
                    in_L=self.max_len,
                    in_D=self.c_p,
                    target_rank=self.z_factor_rank,
                    target_inner_dim=self.c_p,
                )
        else:
            assert self.mode in [
                "orig_no_bias",
                "flash_no_bias",
            ], f"Unknown edge embedder type: {self.mode}. Must be 'orig_no_bias', 'orig_2d_bias', 'flash_no_bias', 'flash_1d_bias', or 'flash_2d_factorize_bias'."

        self.edge_embedder = nn.Sequential(
            nn.Linear(total_edge_feats, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.LayerNorm(self.c_p),
        )

    def embed_relpos(self, pos):
        rel_pos = pos[:, :, None] - pos[:, None, :]
        pos_emb = utils.get_index_embedding(rel_pos, self._cfg.feat_dim, max_len=2056)
        return self.linear_relpos(pos_emb)

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return (
            torch.cat(
                [
                    torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
                    torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
                ],
                dim=-1,
            )
            .float()
            .reshape([num_batch, num_res, num_res, -1])
        )

    def forward(self, node_embed, translations, trans_sc, node_mask, edge_embed=None, edge_mask=None):
        """
        node_embed: [B,L,D]
        translations: [B,L,3]
        trans_sc: [B,L,3]
        node_mask: [B,L]
        edge_embed: [B,L,L,F] (Optional) If not provided, will be computed from node embeddings and translations
        edge_mask: [B,L,L] (Optional)  If not provided, will be inferred from node_mask
        """

        if self.mode == "orig_no_bias" or self.mode == "flash_no_bias":
            # No 2D bias
            assert (
                edge_embed is None and edge_mask is None
            ), "edge_embed and edge_mask must be None for orig_no_bias and flash_no_bias modes"
            edge_embed, z_factor_1, z_factor_2, edge_mask = None, None, None, None
        elif self.mode == "orig_2d_bias":
            if edge_mask is None:
                # We infer edge mask from node mask
                edge_mask = node_mask[:, None] * node_mask[:, :, None]
            if edge_embed is None:
                # We compute edge embeddings from node embeddings and translations
                edge_embed = self.fwd_2d(node_embed, translations, trans_sc, edge_mask)  # 2d mode
            edge_embed = edge_embed * edge_mask[..., None]
            z_factor_1, z_factor_2 = None, None
        elif self.mode == "flash_1d_bias":
            # 1D bias option never materializes the 2D embeddings matrix
            assert edge_embed is None and edge_mask is None, "edge_embed and edge_mask must be None for flash_1d_bias mode"
            z_factor_1, z_factor_2 = self.fwd_1d(node_embed, translations, trans_sc, node_mask)  # 1d mode
            edge_embed, edge_mask = None, None
        elif self.mode == "flash_2d_factorize_bias":
            # 2D bias option materializes the 2D embeddings matrix, either from node embeddings and translations or from edge embeddings
            if edge_mask is None:
                edge_mask = node_mask[:, None] * node_mask[:, :, None]
            if edge_embed is None:
                edge_embed = self.fwd_2d(node_embed, translations, trans_sc, edge_mask)  # 2d mode
            z_factor_1, z_factor_2 = self.factorizer(edge_embed)
            z_factor_1 = rearrange(z_factor_1, "(b d) n r -> b n r d", b=node_embed.shape[0])
            z_factor_2 = rearrange(z_factor_2, "(b d) n r -> b n r d", b=node_embed.shape[0])
            edge_embed, edge_mask = None, None
        else:
            raise ValueError(
                f"Unknown edge embedder type: {self.mode}. Must be 'orig_no_bias', 'orig_2d_bias', 'flash_no_bias', 'flash_1d_bias', or 'flash_2d_factorize_bias'."
            )

        return edge_embed, z_factor_1, z_factor_2, edge_mask

    def fwd_1d(self, s, t, sc_t, p_mask):
        """
        returns [B,L,R,D]
        """

        num_batch, num_res, _ = s.shape
        p_i = self.linear_s_p(s)  # B, L, 2 * R * D
        p_i = rearrange(p_i, "b l (n r d) -> b n l r d", r=self.z_factor_rank, n=2)  # B, 2, L, R, D

        pos = torch.arange(num_res, device=s.device).unsqueeze(0).repeat(num_batch, 1)
        pos_emb = utils.get_index_embedding(pos, self._cfg.feat_dim, max_len=2056)
        pos_emb = self.linear_relpos(pos_emb)  # B, L, 2* R * 2 * D
        pos_emb = rearrange(pos_emb, "b l (n r d) -> b n l r d", r=self.z_factor_rank, n=2)  # B, 2, L, R, 2D

        t_distogram, t_indices = utils.calc_distogram_knn(t, k=self.k, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)
        t_idx_emb = utils.get_index_embedding(rearrange(t_indices, "b l k -> (b l) k"), self._cfg.feat_dim, max_len=2056)
        t_idx_emb = rearrange(t_idx_emb, "(b l) k d -> b l k d", b=num_batch)
        t_cat_emb = torch.cat([t_distogram, t_idx_emb], dim=-1)
        t_embed = self.linear_t_distogram(rearrange(t_cat_emb, "b l k d -> b l (k d)"))  # B, L, num_bins * R
        t_embed = rearrange(t_embed, "b l (n r d) -> b n l r d", r=self.z_factor_rank, n=2)  # B, 2, L, R, num_bins

        sc_distogram, sc_indices = utils.calc_distogram_knn(
            sc_t, k=self.k, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins
        )
        sc_idx_emb = utils.get_index_embedding(rearrange(sc_indices, "b l k -> (b l) k"), self._cfg.feat_dim, max_len=2056)
        sc_idx_emb = rearrange(sc_idx_emb, "(b l) k d -> b l k d", b=num_batch)
        sc_cat_emb = torch.cat([sc_distogram, sc_idx_emb], dim=-1)
        sc_embed = self.linear_sc_distogram(rearrange(sc_cat_emb, "b l k d -> b l (k d)"))  # B, L, num_bins * R
        sc_embed = rearrange(sc_embed, "b l (n r d) -> b n l r d", r=self.z_factor_rank, n=2)  # B, 2, L, R, num_bins

        all_edge_feats = torch.concat([p_i, pos_emb, t_embed, sc_embed], dim=-1)
        edge_feats = self.edge_embedder(all_edge_feats)
        # edge_feats *= p_mask[:, None, :, None, None] #Do we need this?
        z_factor_1 = edge_feats[:, 0, :, :, :]
        z_factor_2 = edge_feats[:, 1, :, :, :]

        return z_factor_1, z_factor_2

    def fwd_2d(self, s, t, sc_t, p_mask):
        """
        s: [B,L,D]
        t: [B,L,3]
        sc_t: [B,L,3]
        p_mask: [B,L,L]
        """

        # raise ValueError(s.shape, t.shape, sc_t.shape, p_mask.shape)
        num_batch, num_res, _ = s.shape
        p_i = self.linear_s_p(s)  # B, L, D
        cross_node_feats = self._cross_concat(
            p_i, num_batch, num_res
        )  # B, L, L, 2*D. Cross concat is similar in spirit to outer product with self.

        pos = torch.arange(num_res, device=s.device).unsqueeze(0).repeat(num_batch, 1)  # B, L. 1D positional encoding
        relpos_feats = self.embed_relpos(pos)  # B, L, L, D. 2D positional encoding

        dist_feats = utils.calc_distogram(t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)  # B, L, L, D
        sc_feats = utils.calc_distogram(sc_t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)  # B, L, L, D

        all_edge_feats = torch.concat([cross_node_feats, relpos_feats, dist_feats, sc_feats], dim=-1)
        edge_feats = self.edge_embedder(all_edge_feats)
        edge_feats *= p_mask.unsqueeze(-1)
        return edge_feats
