import math
from typing import Optional

import torch
from torch import Tensor
from torch.nn import TransformerEncoderLayer, functional as F
from .stochastic_multihead_attention import StochasticMultiheadAttention


class StoTrans(TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None, noise=False) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first, device, dtype)
        self.attention_weights: Optional[Tensor] = None
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.self_attn = StochasticMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                      noise=noise, **factory_kwargs)

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        if x.dim() == 2:
            row_count = int(math.sqrt(x.size(1)))
            x = torch.reshape(x, (x.size(0), row_count, row_count))
        x, weights = self.self_attn(x, x, x,
                                    attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask,
                                    need_weights=True)
        self.attention_weights = weights
        return self.dropout1(x)

    def get_attention_weights(self) -> Optional[Tensor]:
        return self.attention_weights
