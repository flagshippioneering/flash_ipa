# FlashIPA 

Official implementation of FlashIPA, which enhances the efficiency of the IPA module. Our module **reduces training and inference time** and **memory requirements** of standard models.

![scalling](img/scaling.jpg)

## How to use FlashIPA?

After following the setup guide, FlashIPA can be integrated into any model using the IPA module by replacing any original IPA layer with our implementation. The primary input difference from the standard IPA module is the **z_factor**, which represents a memory-efficient graph edge embedding. A complete example of an IPA model is provided in [model.py](src/flash_ipa/model.py), including the full computation of the **z_factor** using the [EdgeEmbedder](src/flash_ipa/edge_embedder.py).


### FlashIPA Model
```python
from flash_ipa.ipa import IPAConfig
from flash_ipa.edge_embedder import EdgeEmbedderConfig
from flash_ipa.model import Model, ModelConfig
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 1  # batch size
max_len = 64  # max length of the input sequence
node_embed_size = 256
edge_embed_size = 128

# IPA config options
use_flash_attn = True  # True for flash ipa, False for original ipa
attn_dtype = "bf16"  # "fp16", "bf16", "fp32". For flash ipa, bf16 or fp16. For original, fp32.
c_s = node_embed_size  # size of the node embedding
c_z = edge_embed_size  # size of the edge embedding
c_hidden = 128  # size of the hidden layer in the ipa
no_heads = 8  # number of attention heads in the ipa
z_factor_rank = 2  # rank of the z factor rank for the edge embedding factorization
no_qk_points = 8  # number of query/key points
no_v_points = 12  # number of value points
seq_tfmr_num_heads = 4  # number of heads in the node embedding update transformer
seq_tfmr_num_layers = 2  # number of layers in the node embedding update transformer
num_blocks = 6  # number of blocks in the model

# Edge embedder config
mode = "flash_1d_bias"  # "orig_no_bias", "orig_2d_bias", "flash_no_bias", "flash_1d_bias", "flash_2d_factorize_bias"
k_nearest_neighbors = 10  # Number of nearest neighbors for the distogram


ipa_conf = IPAConfig(
    use_flash_attn=use_flash_attn,
    attn_dtype=attn_dtype,
    c_s=c_s,
    c_z=c_z,
    c_hidden=c_hidden,
    no_heads=no_heads,
    z_factor_rank=z_factor_rank,  # Rank of the factorization of the edge embedding
    no_qk_points=no_qk_points,
    no_v_points=no_v_points,
    seq_tfmr_num_heads=seq_tfmr_num_heads,
    seq_tfmr_num_layers=seq_tfmr_num_layers,
    num_blocks=num_blocks,
)
edge_features_conf = EdgeEmbedderConfig(
    z_factor_rank=z_factor_rank,  # Rank of the factorization of the edge embedding
    c_s=c_s,  # Size of the node embedding
    c_p=c_z,  # Size of the edge embedding
    mode=mode,  # "orig_no_bias", "orig_2d_bias", "flash_no_bias", "flash_1d_bias", "flash_2d_factorize_bias"
    k=k_nearest_neighbors,  # Number of nearest neighbors for the distogram
    max_len=max_len,  # Maximum length of the input sequence
)
model_conf = ModelConfig(
    mode=mode,
    node_embed_size=node_embed_size,
    edge_embed_size=edge_embed_size,
    ipa=ipa_conf,
    edge_features=edge_features_conf,
)
model = Model(model_conf)
model.to(DEVICE)
batch = {
    "node_embeddings": torch.rand(batch_size, max_len, node_embed_size).to(DEVICE),
    "translations": torch.rand(batch_size, max_len, 3).to(DEVICE),
    "rotations": torch.rand(batch_size, max_len, 3, 3).to(DEVICE),
    "node_mask": torch.ones(batch_size, max_len).to(DEVICE),
}
output = model(batch)
```


### FlashIPA
Example of FlashIPA. If you don't have pre-computed edge embeddings, you can leverage the "flash_1d_bias" mode that computes edge embeddings from the node embeddings without materializing any 2D matrices. If you already have edge embeddings, you can use flashIPA with 2D bias factorization by changing mode="flash_2d_factorize_bias".
```python
import torch

from flash_ipa.ipa import InvariantPointAttention, IPAConfig
from flash_ipa.edge_embedder import EdgeEmbedder, EdgeEmbedderConfig
from flash_ipa.utils import ANG_TO_NM_SCALE
from flash_ipa.rigid import create_rigid

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 1  # batch size
max_len = 64  # max length of the input sequence

# IPA config options
use_flash_attn = True  # True for flash ipa, False for original ipa
attn_dtype = "bf16"  # "fp16", "bf16", "fp32". For flash ipa, bf16 or fp16. For original, fp32.
c_s = 256  # size of the node embedding
c_z = 128  # size of the edge embedding
c_hidden = 128  # size of the hidden layer in the ipa
no_heads = 8  # number of attention heads in the ipa
z_factor_rank = 2  # rank of the z factor rank for the edge embedding factorization
no_qk_points = 8  # number of query/key points
no_v_points = 12  # number of value points

# Edge embedder config
mode = "flash_1d_bias"  # "orig_no_bias", "orig_2d_bias", "flash_no_bias", "flash_1d_bias", "flash_2d_factorize_bias"
k_nearest_neighbors = 10  # Number of nearest neighbors for the distogram

# Create the IPA config
ipa_conf = IPAConfig(
    use_flash_attn=use_flash_attn,
    attn_dtype=attn_dtype,
    c_s=c_s,
    c_z=c_z,
    c_hidden=c_hidden,
    no_heads=no_heads,
    z_factor_rank=z_factor_rank,  # Rank of the factorization of the edge embedding
    no_qk_points=no_qk_points,
    no_v_points=no_v_points,
)
edge_features_conf = EdgeEmbedderConfig(
    z_factor_rank=z_factor_rank,  # Rank of the factorization of the edge embedding
    c_s=c_s,  # Size of the node embedding
    c_p=c_z,  # Size of the edge embedding
    mode=mode,  # "orig_no_bias", "orig_2d_bias", "flash_no_bias", "flash_1d_bias", "flash_2d_factorize_bias"
    k=k_nearest_neighbors,  # Number of nearest neighbors for the distogram
    max_len=max_len,  # Maximum length of the input sequence
)
# Create the IPA layer
ipa_layer = InvariantPointAttention(ipa_conf)
ipa_layer = ipa_layer.to(DEVICE)
rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * ANG_TO_NM_SCALE)
# Create the edge embedder layer, that computes the edge embeddings from the node embeddings and translations
edge_embedder = EdgeEmbedder(edge_features_conf)
edge_embedder = edge_embedder.to(DEVICE)

# Create random inputs
node_embed = torch.randn(batch_size, max_len, c_s).to(DEVICE)
node_mask = torch.ones_like(node_embed[..., 0])
node_embed = node_embed * node_mask[..., None]
translations = torch.randn(batch_size, max_len, 3).to(DEVICE)
rotations = torch.randn(batch_size, max_len, 3, 3).to(DEVICE)
trans_sc = torch.zeros_like(translations)
edge_embed = None # If you have edge embeddings, you can feed them to the edge embedder for factorization
edge_mask = None # If you have an edge mask, you can pass it to the edge embedder.

# Compute edge embeddings from node embeddings and translations
edge_embed, z_factor_1, z_factor_2, edge_mask = edge_embedder(
    node_embed,
    translations,
    trans_sc,
    node_mask,
    edge_embed,
    edge_mask,
)

# Initial rigids
curr_rigids = create_rigid(rotations, translations)
curr_rigids = rigids_ang_to_nm(curr_rigids)

# Apply the IPA layer
# Output of a newly initialized IPA layer is all zeros
ipa_embed = ipa_layer(node_embed, edge_embed, z_factor_1, z_factor_2, curr_rigids, node_mask)
```

### Original IPA
We also provide the implementation of the regular IPA. To use this option, set use_flash_attn=Fasle, attn_dtype="fp32", and mode="orig_2d_bias"
```python
import torch

from flash_ipa.ipa import InvariantPointAttention, IPAConfig
from flash_ipa.edge_embedder import EdgeEmbedder, EdgeEmbedderConfig
from flash_ipa.utils import ANG_TO_NM_SCALE
from flash_ipa.rigid import create_rigid

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 1  # batch size
max_len = 64  # max length of the input sequence

# IPA config options
use_flash_attn = False  # True for flash ipa, False for original ipa
attn_dtype = "fp32"  # "fp16", "bf16", "fp32". For flash ipa, bf16 or fp16. For original, fp32.
c_s = 256  # size of the node embedding
c_z = 128  # size of the edge embedding
c_hidden = 128  # size of the hidden layer in the ipa
no_heads = 8  # number of attention heads in the ipa
z_factor_rank = 2  # rank of the z factor rank for the edge embedding factorization
no_qk_points = 8  # number of query/key points
no_v_points = 12  # number of value points

# Edge embedder config
mode = "orig_2d_bias"  # "orig_no_bias", "orig_2d_bias", "flash_no_bias", "flash_1d_bias", "flash_2d_factorize_bias"

# Create the IPA config
ipa_conf = IPAConfig(
    use_flash_attn=use_flash_attn,
    attn_dtype=attn_dtype,
    c_s=c_s,
    c_z=c_z,
    c_hidden=c_hidden,
    no_heads=no_heads,
    z_factor_rank=z_factor_rank,  # Rank of the factorization of the edge embedding
    no_qk_points=no_qk_points,
    no_v_points=no_v_points,
)
edge_features_conf = EdgeEmbedderConfig(
    mode=mode,  # "orig_no_bias", "orig_2d_bias", "flash_no_bias", "flash_1d_bias", "flash_2d_factorize_bias"
)
# Create the IPA layer
ipa_layer = InvariantPointAttention(ipa_conf)
ipa_layer = ipa_layer.to(DEVICE)
rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * ANG_TO_NM_SCALE)
# Create the edge embedder layer, that computes the edge embeddings from the node embeddings and translations
edge_embedder = EdgeEmbedder(edge_features_conf)
edge_embedder = edge_embedder.to(DEVICE)

# Create random inputs
node_embed = torch.randn(batch_size, max_len, c_s).to(DEVICE)
node_mask = torch.ones_like(node_embed[..., 0])
node_embed = node_embed * node_mask[..., None]
translations = torch.randn(batch_size, max_len, 3).to(DEVICE)
rotations = torch.randn(batch_size, max_len, 3, 3).to(DEVICE)
trans_sc = torch.zeros_like(translations)
edge_embed = None  # Not used for flash ipa
edge_mask = None

# Compute edge embeddings from node embeddings and translations
edge_embed, z_factor_1, z_factor_2, edge_mask = edge_embedder(
    node_embed,
    translations,
    trans_sc,
    node_mask,
    edge_embed,
    edge_mask,
)

# Initial rigids
curr_rigids = create_rigid(rotations, translations)
curr_rigids = rigids_ang_to_nm(curr_rigids)

# Apply the IPA layer
# Output of a newly initialized IPA layer is all zeros
ipa_embed = ipa_layer(node_embed, edge_embed, z_factor_1, z_factor_2, curr_rigids, node_mask)
```

## Setup Guide

To manage environments efficiently, we use [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer). It simplifies managing dependencies and executing scripts.

### As a python package in your uv environement
```bash
uv add "flash_ipa @ git+https://github.com/anonymous/flash_ipa"
```

### For developement
```bash
git clone https://github.com/anonymous/flash_ipa
cd flash_ipa
uv sync
```


## License

This project is licensed under MIT License. See [LICENSE](LICENSE.txt) for more details.

## Citation

``` bash 
@article{liu2025flashipa,
  title={Flash Invariant Point Attention},
  author={Liu, Andrew and Elaldi, Axel and Franklin, Nicholas T and Russell, Nathan and Atwal, Gurinder S and Ban, Yih-En A and Viessmann, Olivia},
  journal={arXiv preprint arXiv:2505.11580},
  year={2025},
  url={https://arxiv.org/abs/2505.11580}
}
```
