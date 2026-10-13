"""Decoder-only transformer in the Llama style: RMSNorm, rotary position embeddings, SwiGLU,
grouped-query attention, tied input/output embeddings, and a KV cache for generation.

Compared with the nanoGPT model from the course:

| nanoGPT            | here                   | why                                                         |
|--------------------|------------------------|-------------------------------------------------------------|
| LayerNorm          | RMSNorm                | no mean subtraction or bias; cheaper, works as well          |
| learned positions  | RoPE                   | relative positions, nothing to learn, extends past training  |
| GELU MLP (4d)      | SwiGLU (≈ 8/3 d)       | gated MLP, better quality at the same parameter count        |
| multi-head attn    | grouped-query attn     | fewer K/V heads → smaller KV cache, faster generation        |
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GPTConfig:
    """Model size and shape. Saved with every checkpoint (`to_dict`) so a run can be loaded without guessing."""
    vocab_size: int = 8192
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    dropout: float = 0.0
    abs_pos: bool = False        # also add a learned embedding of the absolute position (GPT-2 style)

    @property
    def head_dim(self) -> int:
        """Width of one attention head."""
        return self.d_model // self.n_heads

    @property
    def ffn_dim(self) -> int:
        """Hidden width of the MLP, rounded up to a multiple of 64."""
        # SwiGLU has three matrices instead of two; 8/3·d keeps the parameter count of a 4·d MLP.
        return int(math.ceil(8 * self.d_model / 3 / 64) * 64)

    def to_dict(self) -> dict[str, Any]:
        """The config as a plain dict, as stored in checkpoints."""
        return asdict(self)


class RMSNorm(nn.Module):
    """Scale each vector to unit root-mean-square, then by a learned gain (LayerNorm without the mean)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x / rms(x) × gain, over the last dimension."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_tables(head_dim: int, seq_len: int, theta: float,
                device: torch.device | str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin of the rotation angle for every (position, feature pair), (seq_len, head_dim / 2) each."""
    freqs = 1.0 / theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    angles = torch.outer(torch.arange(seq_len, device=device).float(), freqs)   # (T, head_dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each (even, odd) pair of features by a position-dependent angle. x: (B, H, T, D)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    """Causal self-attention with rotary positions and grouped-query heads (n_kv_heads ≤ n_heads)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.kv = nn.Linear(cfg.d_model, 2 * cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.out = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                cache: dict[str, torch.Tensor | int] | None = None) -> torch.Tensor:
        """x (B, T, d) → (B, T, d). With a cache (generation), K/V of earlier steps are reused."""
        B, T, _ = x.shape
        c = self.cfg
        q = self.q(x).view(B, T, c.n_heads, c.head_dim).transpose(1, 2)
        k, v = self.kv(x).view(B, T, 2, c.n_kv_heads, c.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:                    # generation: add to what earlier steps saw
            k, v = self._extend(cache, k, v)
            if T == 1 and "size" in cache and c.n_kv_heads != c.n_heads:
                return self._one_query(q, k, v)
        if c.n_kv_heads != c.n_heads:            # each K/V head serves a group of query heads
            k = k.repeat_interleave(c.n_heads // c.n_kv_heads, dim=1)
            v = v.repeat_interleave(c.n_heads // c.n_kv_heads, dim=1)
        # With a cache and T == 1 the single query may attend to everything before it.
        causal = cache is None or T > 1
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal,
                                           dropout_p=c.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(B, T, -1))

    @staticmethod
    def _extend(cache: dict[str, torch.Tensor | int], k: torch.Tensor,
                v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """K/V of every step so far. With cache["size"] (generate(preallocate=True)) each step writes into
        buffers of that length; otherwise the cache grows by concatenation, which copies all of it every step:
        fine for a few hundred tokens, far too slow for thousands (brushgpt: 3,046 tokens a painting)."""
        if "size" not in cache:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
            return k, v
        if "k" not in cache:
            B, H, _, D = k.shape
            cache["k"], cache["v"], cache["n"] = k.new_empty(B, H, cache["size"], D), v.new_empty(B, H, cache["size"], D), 0
        n, T = cache["n"], k.size(2)
        cache["k"][:, :, n:n + T] = k
        cache["v"][:, :, n:n + T] = v
        cache["n"] = n + T
        return cache["k"][:, :, :n + T], cache["v"][:, :, :n + T]

    def _one_query(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Attention for a single new token: each K/V head's group of query heads is laid along the query axis
        instead of repeating K/V, which would copy the cache again."""
        c, B = self.cfg, q.size(0)
        y = F.scaled_dot_product_attention(q.reshape(B, c.n_kv_heads, c.n_heads // c.n_kv_heads, c.head_dim), k, v)
        return self.out(y.reshape(B, 1, -1))


class SwiGLU(nn.Module):
    """Gated MLP: down(silu(gate(x)) · up(x))."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """down(silu(gate(x)) × up(x)): the gate decides how much of each hidden unit passes."""
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """One pre-norm transformer layer: attention, then the MLP, each added to the residual stream."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                cache: dict[str, torch.Tensor | int] | None = None) -> torch.Tensor:
        """x + attention(norm(x)), then + MLP(norm(x)); `cache` holds this layer's keys and values when generating."""
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    """The whole decoder: token embedding, the blocks, a final norm and the (tied) output layer."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # RoPE only tells attention how far apart two tokens are. Where the meaning of a token depends on
        # its absolute slot (linegpt: every 8 tokens are x0 y0 … y3 of one stroke), a learned position
        # embedding gives that slot directly.
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model) if cfg.abs_pos else None
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight     # tied: one table to read and write tokens
        cos, sin = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():  # residual outputs start small (GPT-2 trick)
            if name.endswith("out.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m: nn.Module) -> None:
        """Every linear and embedding weight starts from N(0, 0.02²) (GPT-2)."""
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self) -> int:
        """Parameter count (the tied embedding and output table counted once)."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                loss_mask: torch.Tensor | None = None, cache: list[dict[str, torch.Tensor | int]] | None = None,
                start_pos: int = 0) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx (B, T) → logits (B, T, V). With targets, also mean cross-entropy over positions
        where loss_mask is 1 (all positions when no mask is given)."""
        T = idx.size(1)
        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]
        x = self.embed(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(start_pos, start_pos + T, device=idx.device))
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, None if cache is None else cache[i])
        logits = self.head(self.norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
        if loss_mask is None:
            return logits, loss.mean()
        mask = loss_mask.reshape(-1).float()
        return logits, (loss * mask).sum() / mask.sum().clamp(min=1)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, stop_id: int | None = None,
                 uncond: torch.Tensor | None = None, guidance: float = 1.0,
                 preallocate: bool = False) -> torch.Tensor:
        """Autoregressive sampling with a KV cache; temperature 0 means greedy.

        Classifier-free guidance (Ho & Salimans 2021): `uncond` is the same prompt with the condition
        replaced by a "no condition" token, run in the same batch. The next-token logits become
        uncond + guidance × (cond − uncond), so guidance 1 is plain sampling and larger values push
        towards what the condition adds. Both prompts are extended with the token sampled for the pair.

        preallocate: hold the KV cache in buffers of the final length instead of growing it by concatenation
        (same tokens; needed for long sequences, see Attention.forward)."""
        self.eval()
        n = idx.size(0)
        guided = uncond is not None
        if guided:
            assert uncond.shape == idx.shape, "the unconditional prompt must have the prompt's shape"
        size = min(self.cfg.max_seq_len, idx.size(1) + max_new_tokens)
        cache = [{"size": size} if preallocate else {} for _ in self.blocks]
        logits, _ = self(torch.cat([idx, uncond]) if guided else idx, cache=cache)
        pos = idx.size(1)
        out = idx
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :]
            if guided:
                cond, free = next_logits[:n], next_logits[n:]
                next_logits = free + guidance * (cond - free)
            nxt = _pick(next_logits, temperature, top_k)
            out = torch.cat([out, nxt], dim=1)
            if (stop_id is not None and (nxt == stop_id).all()) or pos >= self.cfg.max_seq_len:
                break
            logits, _ = self(torch.cat([nxt, nxt]) if guided else nxt, cache=cache, start_pos=pos)
            pos += 1
        return out


def _pick(logits: torch.Tensor, temperature: float, top_k: int | None) -> torch.Tensor:
    """One token per row: the most likely at temperature 0, otherwise sampled from the temperature-scaled
    distribution, optionally cut to the top_k tokens. (B, V) → (B, 1)."""
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k:
        kth = logits.topk(top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    return torch.multinomial(logits.softmax(-1), 1)


def grow_vocab(model: GPT, vocab_size: int) -> GPT:
    """A copy of `model` with room for vocab_size tokens (design-v2 §6.4). Old rows of the tied
    embedding are kept, so the logits of every old token are unchanged; new rows start like the
    model's own initialisation (normal, std 0.02) and are learned in fine-tuning."""
    if vocab_size < model.cfg.vocab_size:
        raise ValueError("a vocabulary can only grow")
    cfg = GPTConfig(**{**model.cfg.to_dict(), "vocab_size": vocab_size})
    grown = GPT(cfg).to(next(model.parameters()).device)
    state = model.state_dict()
    old = state["embed.weight"]
    table = grown.embed.weight.detach().clone()
    table[:len(old)] = old
    state["embed.weight"] = state["head.weight"] = table
    grown.load_state_dict(state)
    return grown
