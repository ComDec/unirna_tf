from typing import Optional

import torch
from einops import rearrange
from flash_attn.bert_padding import unpad_input
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from torch import Tensor


def unirna_flash_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bsz: int,
    num_heads: int,
    seqlen: int,
    head_dim: int,
    attn_dropout: float,
    key_padding_mask: Optional[Tensor] = None,
) -> Tensor:
    """Compute FlashAttention with optional padding mask for UniRNA tensors."""
    q = q.reshape(bsz, num_heads, seqlen, head_dim).transpose(1, 2).reshape(bsz, seqlen, num_heads * head_dim)
    k = k.reshape(bsz, num_heads, seqlen, head_dim).transpose(1, 2).reshape(bsz, seqlen, num_heads * head_dim)
    v = v.reshape(bsz, num_heads, seqlen, head_dim).transpose(1, 2).reshape(bsz, seqlen, num_heads * head_dim)

    q_unpad = rearrange(q, "b s (h d) -> (b s) h d", h=num_heads)
    cu_seqlens_q = torch.arange(0, (bsz + 1) * seqlen, step=seqlen, dtype=torch.int32, device=q_unpad.device)
    max_seqlen_q = seqlen

    if key_padding_mask is not None:
        if key_padding_mask.dtype is not torch.bool:
            key_padding_mask = key_padding_mask.bool()
        k_unpad, _, cu_seqlens_k, max_seqlen_k, seq_used = unpad_input(k, key_padding_mask)
        k_unpad = rearrange(k_unpad, "nnz (h d) -> nnz h d", h=num_heads)
        v_unpad, _, _, _, seq_used = unpad_input(v, key_padding_mask)
        v_unpad = rearrange(v_unpad, "nnz (h d) -> nnz h d", h=num_heads)

        # there is no need to conduct unpad_input for q
        # q_unpad_real, indices_q_real, cu_seqlens_q_real, max_seqlen_q_real = unpad_input(q, key_padding_mask)
        # q_unpad_real = rearrange(q_unpad_real, "nnz (h d) -> nnz h d", h=num_heads)
    else:
        k_unpad = rearrange(k, "b s (h d) -> (b s) h d", h=num_heads)
        v_unpad = rearrange(v, "b s (h d) -> (b s) h d", h=num_heads)
        cu_seqlens_k = torch.arange(0, (bsz + 1) * seqlen, step=seqlen, dtype=torch.int32, device=q_unpad.device)
        max_seqlen_k = seqlen

    o: Tensor = flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=attn_dropout,
        softmax_scale=1.0,
        causal=False,
    )

    o = o.reshape(bsz, seqlen, num_heads, head_dim)
    return o
