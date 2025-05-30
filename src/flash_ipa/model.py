"""
Example of neural network architecture.

Code adapted from
https://github.com/microsoft/protein-frame-flow/blob/main/models/flow_model.py
"""

import torch
from torch import nn

from flash_ipa.edge_embedder import EdgeEmbedder, EdgeEmbedderConfig
from flash_ipa.ipa import StructureModuleTransition, BackboneUpdate, EdgeTransition, InvariantPointAttention, IPAConfig
from flash_ipa.utils import ANG_TO_NM_SCALE, NM_TO_ANG_SCALE, check_config_ipa
from flash_ipa.rigid import create_rigid
from flash_ipa.linear import Linear
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # 5 options: "orig_no_bias", "orig_2d_bias", "flash_no_bias", "flash_1d_bias", "flash_2d_factorize_bias"
    mode: str = "flash_1d_bias"
    node_embed_size: int = 256
    edge_embed_size: int = 128
    ipa: IPAConfig = field(default_factory=IPAConfig)
    edge_features: EdgeEmbedderConfig = field(default_factory=EdgeEmbedderConfig)


class Model(nn.Module):

    def __init__(self, model_conf):
        super(Model, self).__init__()
        # Check variables are consistent for experiment.
        check_config_ipa(model_conf.ipa, model_conf.mode)

        # Model config
        self._model_conf = model_conf
        self.mode = model_conf.mode
        self._ipa_conf = model_conf.ipa

        # Rigid transforms utils
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * NM_TO_ANG_SCALE)

        # Edge embedder
        self.edge_embedder = EdgeEmbedder(self._model_conf.edge_features)

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            # IPA block
            self.trunk[f"ipa_{b}"] = InvariantPointAttention(self._ipa_conf)

            # Node embedding update
            self.trunk[f"ipa_ln_{b}"] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=self._ipa_conf.c_s,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=self._ipa_conf.c_s,
                batch_first=True,
                dropout=0.0,
                norm_first=False,
            )
            self.trunk[f"seq_tfmr_{b}"] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False
            )
            self.trunk[f"post_tfmr_{b}"] = Linear(self._ipa_conf.c_s, self._ipa_conf.c_s, init="final")
            self.trunk[f"node_transition_{b}"] = StructureModuleTransition(c=self._ipa_conf.c_s)

            # Frame update
            self.trunk[f"bb_update_{b}"] = BackboneUpdate(self._ipa_conf.c_s, use_rot_updates=True)

            # Edge embedding update
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
            trans_sc: (B, L, 3) (Optional) If not provided, will be set to 0
            node_mask: (B, L) (Optional) If not provided, will be set to 1
            edge_embeddings: (B, L, L, F) (Optional) If not provided, will be computed from node embeddings and translations
            edge_mask: (B, L, L) (Optional) If not provided, either infered from node_mask or set to 1
        """
        # Inputs
        init_node_embed = input_feats["node_embeddings"]
        translations = input_feats["translations"]
        rotations = input_feats["rotations"]
        if "trans_sc" not in input_feats:
            trans_sc = torch.zeros_like(translations)
        else:
            trans_sc = input_feats["trans_sc"]
        if "edge_embeddings" in input_feats:
            init_edge_embed = input_feats["edge_embeddings"]
        else:
            init_edge_embed = None

        # Masks
        if "node_mask" not in input_feats:
            node_mask = torch.ones_like(init_node_embed[..., 0])
        else:
            node_mask = input_feats["node_mask"]
        if "edge_mask" in input_feats:
            edge_mask = input_feats["edge_mask"]
        else:
            edge_mask = None

        # Apply node mask
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]

        # Compute edge embeddings from node embeddings and translations
        edge_embed, z_factor_1, z_factor_2, edge_mask = self.edge_embedder(
            init_node_embed,
            translations,
            trans_sc,
            node_mask,
            init_edge_embed,
            edge_mask,
        )

        # Initial rigids
        curr_rigids = create_rigid(
            rotations,
            translations,
        )
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)

        # Main trunk
        for b in range(self._ipa_conf.num_blocks):
            # Compute IPA embeddings
            ipa_embed = self.trunk[f"ipa_{b}"](
                node_embed,
                edge_embed,  # Should be None for flash attn mode
                z_factor_1,  # Should be None for not flash attn mode
                z_factor_2,  # Should be None for not flash attn mode
                curr_rigids,
                node_mask,
            )
            ipa_embed *= node_mask[..., None]

            # Update embeddings
            node_embed = self.trunk[f"ipa_ln_{b}"](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f"seq_tfmr_{b}"](node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f"post_tfmr_{b}"](seq_tfmr_out)
            node_embed = self.trunk[f"node_transition_{b}"](node_embed)
            node_embed = node_embed * node_mask[..., None]

            # Update frames
            rigid_update = self.trunk[f"bb_update_{b}"](node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(rigid_update, node_mask[..., None])

            # Update edge embeddings
            if b < self._ipa_conf.num_blocks - 1:
                if self.mode == "orig_2d_bias":
                    edge_embed = self.trunk[f"edge_transition_{b}"](node_embed, edge_embed)  # edge_embed is B,L,L,D
                    edge_embed *= edge_mask[..., None]
                elif self.mode == "flash_1d_bias" or self.mode == "flash_2d_factorize_bias":
                    z_factor_1, z_factor_2 = self.trunk[f"edge_transition_{b}"](node_embed, None, z_factor_1, z_factor_2)
                    z_factor_1 *= node_mask[:, :, None, None]
                    z_factor_2 *= node_mask[:, :, None, None]
                else:
                    # no 2D bias
                    continue

        # Convert back to angstroms
        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()

        # Outputs
        return {
            "pred_trans": pred_trans,
            "pred_rotmats": pred_rotmats,
        }
