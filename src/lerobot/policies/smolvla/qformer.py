from __future__ import annotations
import torch
from torch import nn


class _QFormerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        kv_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_self_attn: bool = True,
    ) -> None:
        super().__init__()
        self.use_self_attn = use_self_attn

        # Project image (KV) features into the Q-Former hidden dim once per
        # block so attention math stays consistent regardless of SigLIP width.
        self.kv_proj = nn.Linear(kv_dim, hidden_dim, bias=True)
        self.kv_norm = nn.LayerNorm(hidden_dim)

        self.cross_attn_norm_q = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_attn_norm_q_2 = nn.LayerNorm(hidden_dim)
        self.cross_attn_2 = nn.MultiheadAttention(
            embed_dim = hidden_dim,
            num_heads = num_heads,
            dropout = dropout,
            batch_first = True,
        )

        self.ffn_norm = nn.LayerNorm(hidden_dim)
        ffn_hidden = int(hidden_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        kv: torch.Tensor,
        kv_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kv_proj = self.kv_norm(self.kv_proj(kv))

        q_norm = self.cross_attn_norm_q(queries)
        attn_out, _ = self.cross_attn(
            query=q_norm,
            key=kv_proj,
            value=kv_proj,
            key_padding_mask=kv_key_padding_mask,
            need_weights=False,
        )
        queries = queries + attn_out
        q_norm = self.self_attn_norm(queries)
        sa_out, _ = self.self_attn(
            query=q_norm,
            key=q_norm,
            value=q_norm,
            need_weights=False,
        )
        queries = queries + sa_out

        q_norm = self.cross_attn_norm_q_2(queries)
        attn_out_2, _ = self.cross_attn_2(
            query=q_norm,
            key=kv_proj,
            value=kv_proj,
            key_padding_mask=kv_key_padding_mask,
            need_weights=False,
        )
        queries = queries + attn_out_2

        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries


class QFormer(nn.Module):
    def __init__(
        self,
        num_queries: int,
        hidden_dim: int,
        kv_dim: int,
        out_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        self_attn_every_n_layers: int = 1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.self_attn_every_n_layers = self_attn_every_n_layers

        self.queries = nn.Parameter(torch.empty(num_queries, hidden_dim))

        self.blocks = nn.ModuleList()
        for layer_idx in range(num_layers):
            use_sa = self_attn_every_n_layers > 0 and (layer_idx % self_attn_every_n_layers == 0)
            self.blocks.append(
                _QFormerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    kv_dim=kv_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    use_self_attn=use_sa,
                )
            )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        # BLIP-2 style: queries ~ N(0, 0.02), Xavier on linears, zero biases,
        # LayerNorm at PyTorch defaults (weight=1, bias=0).
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                if module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.zeros_(module.out_proj.bias)

    def forward(
        self,
        image_tokens: torch.Tensor,
        image_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz = image_tokens.shape[0]
        queries = self.queries.unsqueeze(0).expand(bsz, -1, -1).to(dtype=image_tokens.dtype)
        kv_key_padding_mask = None
        if image_token_mask is not None:
            kv_key_padding_mask = ~image_token_mask.to(dtype=torch.bool)

        for block in self.blocks:
            queries = block(queries, image_tokens, kv_key_padding_mask=kv_key_padding_mask)

        queries = self.final_norm(queries)
        queries = self.out_proj(queries)
        return queries
