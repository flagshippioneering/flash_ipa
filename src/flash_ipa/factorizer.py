import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple
from torch import nn, Tensor
import math


class LinearFactorizer(nn.Module):

    def __init__(self, in_L, in_D, target_rank=4, target_inner_dim=8):
        super().__init__()
        self.target_rank = target_rank
        self.target_inner_dim = target_inner_dim
        # self.linear_col = nn.Linear(in_L * in_D, target_rank * target_inner_dim, bias=False)
        # self.linear_row = nn.Linear(in_L * in_D, target_rank * target_inner_dim, bias=False)
        self.length_compressor = nn.Linear(in_L, target_rank, bias=False)
        self.inner_compressor = nn.Linear(in_D, target_inner_dim, bias=False)
        self.in_L = in_L
        self.length_norm = nn.LayerNorm(target_rank)
        self.inner_norm = nn.LayerNorm(target_inner_dim)

        nn.init.kaiming_uniform_(self.length_compressor.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.inner_compressor.weight, a=math.sqrt(5))

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Input:
            x: (B, L, L, D)
            mask: (B, L), optional length mask
        Output:
            U: (B * target_inner_dim, L, target_rank)
            V: (B * target_inner_dim, L, target_rank)
        """
        # Compress along length
        L_orig = x.shape[1]
        x = F.pad(x, (0, 0, 0, self.in_L - x.shape[2], 0, self.in_L - x.shape[1]), value=0.0)
        row_embed = self.length_compressor(rearrange(x, "B R C D -> B C D R"))[:, :L_orig, :, :]  # (B, L, D, target_rank)
        col_embed = self.length_compressor(rearrange(x, "B R C D -> B R D C"))[:, :L_orig, :, :]  # (B, L, D, target_rank)

        row_embed = self.length_norm(row_embed)
        col_embed = self.length_norm(col_embed)

        row_embed = self.inner_compressor(rearrange(row_embed, "B C D R -> B C R D"))  # (B, L, target_rank, target_inner_dim)
        col_embed = self.inner_compressor(rearrange(col_embed, "B R D C -> B R C D"))  # (B, L, target_rank, target_inner_dim)

        row_embed = self.inner_norm(row_embed)
        col_embed = self.inner_norm(col_embed)
        if mask is not None:
            # Apply mask to row_embed and col_embed
            row_embed = row_embed * mask[:, :, None, None]
            col_embed = col_embed * mask[:, :, None, None]

        row_embed = rearrange(row_embed, "B C R D -> (B D) C R")[:,] / math.sqrt(self.target_rank)  # (B * D, L, target_rank)
        col_embed = rearrange(col_embed, "B R C D -> (B D) R C") / math.sqrt(self.target_rank)  # (B * D, L, target_rank)

        # row_embed = row_embed / self.target_rank
        # col_embed = col_embed / self.target_rank

        return row_embed, col_embed
