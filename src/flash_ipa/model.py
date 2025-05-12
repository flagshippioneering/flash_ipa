"""
Example of neural network architecture.

Code adapted from
https://github.com/microsoft/protein-frame-flow/blob/main/models/flow_model.py
"""

import torch
from torch import nn
from einops import rearrange

from flash_ipa.edge_embedder import EdgeEmbedder, EdgeEmbedderConfig
from flash_ipa.ipa import StructureModuleTransition, BackboneUpdate, EdgeTransition, InvariantPointAttention, IPAConfig
from flash_ipa.utils import ANG_TO_NM_SCALE, NM_TO_ANG_SCALE
from flash_ipa.rigid import create_rigid
from flash_ipa.factorizer import LinearFactorizer
from flash_ipa.linear import Linear
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # 5 options: "orig_no_bias", "orig_2d_bias", "flash_no_bias", "flash_1d_bias", "flash_2d_factorize_bias"
    mode: str = "flash_1d_bias"
    max_len: int = 256
    node_embed_size: int = 256
    edge_embed_size: int = 128
    ipa: IPAConfig = field(default_factory=IPAConfig)
    edge_features: EdgeEmbedderConfig = field(default_factory=EdgeEmbedderConfig)


class Model(nn.Module):

    def __init__(self, model_conf):
        super(Model, self).__init__()
        self._model_conf = model_conf
        self.mode = model_conf.mode
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * NM_TO_ANG_SCALE)
        self.edge_embedder = EdgeEmbedder(model_conf.edge_features, mode="1d" if self.mode == "flash_1d_bias" else "2d")

        """
        Check variables are consistent for experiment.
        """
        if self.mode == "orig_no_bias":
            assert (
                self._ipa_conf.c_z == 0 and self._ipa_conf.z_factor_rank == 0 and self._ipa_conf.use_flash_attn == False
            ), "Expecting self._ipa_conf.c_z == 0 and self._ipa_conf.z_factor_rank == 0 and self._ipa_conf.use_flash_attn == False, but got {self._ipa_conf.c_z}, {self._ipa_conf.z_factor_rank}, {self._ipa_conf.use_flash_attn}."
        elif self.mode == "orig_2d_bias":
            assert (
                self._ipa_conf.c_z > 0 and self._ipa_conf.z_factor_rank == 0 and self._ipa_conf.use_flash_attn == False
            ), "Expecting self._ipa_conf.c_z > 0 and self._ipa_conf.z_factor_rank == 0 and self._ipa_conf.use_flash_attn == False, but got {self._ipa_conf.c_z}, {self._ipa_conf.z_factor_rank}, {self._ipa_conf.use_flash_attn}."
        elif self.mode == "flash_no_bias":
            assert (
                self._ipa_conf.c_z == 0 and self._ipa_conf.z_factor_rank == 0 and self._ipa_conf.use_flash_attn == True
            ), "Expecting self._ipa_conf.c_z == 0 and self._ipa_conf.z_factor_rank == 0 and self._ipa_conf.use_flash_attn == True, but got {self._ipa_conf.c_z}, {self._ipa_conf.z_factor_rank}, {self._ipa_conf.use_flash_attn}."
        elif self.mode == "flash_1d_bias":
            assert (
                self._ipa_conf.c_z > 0 and self._ipa_conf.z_factor_rank > 0 and self._ipa_conf.use_flash_attn == True
            ), "Expecting self._ipa_conf.c_z > 0 and self._ipa_conf.z_factor_rank > 0 and self._ipa_conf.use_flash_attn == True, but got {self._ipa_conf.c_z}, {self._ipa_conf.z_factor_rank}, {self._ipa_conf.use_flash_attn}."
        elif self.mode == "flash_2d_factorize_bias":
            assert (
                self._ipa_conf.c_z > 0 and self._ipa_conf.z_factor_rank > 0 and self._ipa_conf.use_flash_attn == True
            ), "Expecting self._ipa_conf.c_z > 0 and self._ipa_conf.z_factor_rank > 0 and self._ipa_conf.use_flash_attn == True, but got {self._ipa_conf.c_z}, {self._ipa_conf.z_factor_rank}, {self._ipa_conf.use_flash_attn}."
        else:
            raise ValueError(
                f"Invalid mode: {self.mode}. Must be one of ['orig_no_bias', 'orig_2d_bias', 'flash_no_bias', 'flash_1d_bias', 'flash_2d_factorize_bias']."
            )

        if self.mode == "flash_2d_factorize_bias":
            self.factorizer = LinearFactorizer(
                in_L=model_conf.max_len,
                in_D=self._ipa_conf.c_z,
                target_rank=self._ipa_conf.z_factor_rank,
                target_inner_dim=self._ipa_conf.c_z,
            )

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f"ipa_{b}"] = InvariantPointAttention(self._ipa_conf)
            self.trunk[f"ipa_ln_{b}"] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False,
            )
            self.trunk[f"seq_tfmr_{b}"] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False
            )
            self.trunk[f"post_tfmr_{b}"] = Linear(tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f"node_transition_{b}"] = StructureModuleTransition(c=self._ipa_conf.c_s)
            self.trunk[f"bb_update_{b}"] = BackboneUpdate(self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f"edge_transition_{b}"] = EdgeTransition(
                    mode="2d" if self.mode == "orig_2d_bias" else "1d",
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                    z_factor_rank=self._ipa_conf.z_factor_rank,
                )

    def forward(self, input_feats):
        """
        Assuming frames are already computed.
        input_feats:
            node_embeddings: (B, L, D)
            translations: (B, L, 3)
            rotations: (B, L, 3, 3)
            res_mask: (B, L)

        """
        # Masks
        node_mask = input_feats["res_mask"]
        if self.mode in ["orig_2d_bias", "flash_2d_factorize_bias"]:
            # Edge mask exist only if we use 2d bias
            edge_mask = node_mask[:, None] * node_mask[:, :, None]
        else:
            edge_mask = None

        # Inputs
        init_node_embed = input_feats["node_embeddings"]
        translations = input_feats["translations"]
        rotations = input_feats["rotations"]

        if "trans_sc" not in input_feats:
            trans_sc = torch.zeros_like(translations)
        else:
            trans_sc = input_feats["trans_sc"]

        # Initialize edge embeddings depending on the mode
        if self.mode == "orig_no_bias" or self.mode == "flash_no_bias":
            init_edge_embed, z_factor_1, z_factor_2 = None, None, None
        elif self.mode == "orig_2d_bias":
            init_edge_embed = self.edge_embedder(init_node_embed, translations, trans_sc, edge_mask)  # 2d mode
        elif self.mode == "flash_1d_bias":
            z_factor_1, z_factor_2 = self.edge_embedder(init_node_embed, translations, trans_sc, node_mask)  # 1d mode
        elif self.mode == "flash_2d_factorize_bias":
            init_edge_embed = self.edge_embedder(init_node_embed, translations, trans_sc, edge_mask)  # 2d mode
            z_factor_1, z_factor_2 = self.factorizer(init_edge_embed)
            z_factor_1 = rearrange(z_factor_1, "(b d) n r -> b n r d", b=init_node_embed.shape[0])
            z_factor_2 = rearrange(z_factor_2, "(b d) n r -> b n r d", b=init_node_embed.shape[0])

        # Apply masks
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        if self.mode == "orig_2d_bias":
            # The edge_embed is used for slow IPA. Otherwise, use z_factor or node_embed.
            edge_embed = init_edge_embed * edge_mask[..., None]
        else:
            edge_embed = None

        # Initial rigids
        curr_rigids = create_rigid(
            rotations,
            translations,
        )
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)

        # Main trunk
        for b in range(self._ipa_conf.num_blocks):
            if self._ipa_conf.use_flash_attn:
                # The FlashAttention case uses pseudo-factors of the pair bias.
                ipa_embed = self.trunk[f"ipa_{b}"](
                    node_embed,
                    None,
                    z_factor_1,
                    z_factor_2,
                    curr_rigids,
                    mask=node_mask,
                )
            else:
                # The non-FlashAttention case uses the full pair bias (edge_embed).
                ipa_embed = self.trunk[f"ipa_{b}"](node_embed, edge_embed, None, None, curr_rigids, node_mask)

            # Update embedings and frame
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f"ipa_ln_{b}"](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f"seq_tfmr_{b}"](node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f"post_tfmr_{b}"](seq_tfmr_out)
            node_embed = self.trunk[f"node_transition_{b}"](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f"bb_update_{b}"](node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                if self.mode == "orig_2d_bias":
                    edge_embed = self.trunk[f"edge_transition_{b}"](node_embed, edge_embed)  # edge_embed is B,L,L,D
                    edge_embed *= edge_mask[..., None]
                elif self.mode == "flash_1d_bias" or self.mode == "flash_2d_factorize_bias":
                    z_factor_1, z_factor_2 = self.trunk[f"edge_transition_{b}"](node_embed, None, z_factor_1, z_factor_2)
                    z_factor_1 *= node_mask[:, :, None, None]
                    z_factor_2 *= node_mask[:, :, None, None]
                else:
                    # no bias
                    continue

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()

        return {
            "pred_trans": pred_trans,
            "pred_rotmats": pred_rotmats,
        }
