"""GPT model: properties that are easy to break silently."""

import torch

from sellanything_ml.llm.model import GPT, GPTConfig, apply_rope, rope_tables

CFG = GPTConfig(vocab_size=97, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32)


def model():
    torch.manual_seed(0)
    return GPT(CFG).eval()


def test_shapes_and_initial_loss_near_uniform():
    m = model()
    idx = torch.randint(0, CFG.vocab_size, (3, 16))
    # Random targets, not idx: with tied embeddings an untrained model already leans toward
    # predicting its own input token (logit = normed(e_x)·e_x), so targets=idx scores ~3.4.
    targets = torch.randint(0, CFG.vocab_size, (3, 16))
    logits, loss = m(idx, targets=targets)
    assert logits.shape == (3, 16, CFG.vocab_size)
    assert abs(loss.item() - torch.log(torch.tensor(97.0)).item()) < 0.3   # ln(97) ≈ 4.57


def test_causal_future_tokens_do_not_change_the_past():
    m = model()
    a = torch.randint(0, CFG.vocab_size, (1, 12))
    b = a.clone()
    b[0, 8:] = (b[0, 8:] + 1) % CFG.vocab_size
    la, _ = m(a)
    lb, _ = m(b)
    assert torch.allclose(la[0, :8], lb[0, :8], atol=1e-5)
    assert not torch.allclose(la[0, 8:], lb[0, 8:], atol=1e-5)


def test_kv_cache_matches_full_recompute():
    m = model()
    prompt = torch.randint(0, CFG.vocab_size, (1, 5))
    out = m.generate(prompt, max_new_tokens=10, temperature=0)
    # Greedy decoding without the cache: recompute the whole sequence every step.
    seq = prompt
    for _ in range(10):
        logits, _ = m(seq)
        seq = torch.cat([seq, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(out, seq)


def test_preallocated_cache_gives_the_same_tokens():
    m = model()
    prompt = torch.randint(0, CFG.vocab_size, (3, 4))
    assert torch.equal(m.generate(prompt, 20, temperature=0, preallocate=True), m.generate(prompt, 20, temperature=0))
    torch.manual_seed(1)
    grown = m.generate(prompt, 20)
    torch.manual_seed(1)
    assert torch.equal(m.generate(prompt, 20, preallocate=True), grown)


def test_loss_mask_only_counts_masked_positions():
    m = model()
    idx = torch.randint(0, CFG.vocab_size, (2, 10))
    targets = torch.randint(0, CFG.vocab_size, (2, 10))
    mask = torch.zeros(2, 10)
    mask[:, 6:] = 1
    _, masked = m(idx, targets, mask)
    logits, _ = m(idx)
    expected = torch.nn.functional.cross_entropy(logits[:, 6:].reshape(-1, CFG.vocab_size),
                                                 targets[:, 6:].reshape(-1))
    assert torch.allclose(masked, expected, atol=1e-5)


def test_rope_preserves_length_and_relative_angle():
    cos, sin = rope_tables(8, 16, 10000.0)
    x = torch.randn(1, 1, 16, 8)
    r = apply_rope(x, cos, sin)
    assert torch.allclose(r.norm(dim=-1), x.norm(dim=-1), atol=1e-5)    # rotation keeps length
    # dot product of the same vectors placed at (p, p+3) is the same for every p
    q = torch.randn(8)
    k = torch.randn(8)
    qs = apply_rope(q.expand(1, 1, 16, 8), cos, sin)[0, 0]
    ks = apply_rope(k.expand(1, 1, 16, 8), cos, sin)[0, 0]
    dots = [(qs[p] @ ks[p + 3]).item() for p in range(10)]
    assert max(dots) - min(dots) < 1e-4


def test_parameter_count_matches_formula():
    m = model()
    d, f, L, V = CFG.d_model, CFG.ffn_dim, CFG.n_layers, CFG.vocab_size
    kv = CFG.n_kv_heads * CFG.head_dim
    per_block = d * d + d * 2 * kv + d * d + 3 * d * f + 2 * d      # q, kv, out, SwiGLU, 2 norms
    assert m.num_params() == V * d + L * per_block + d               # tied head, final norm


def test_grow_vocab_keeps_every_old_logit():
    from sellanything_ml.llm.model import grow_vocab
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=50, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=16)).eval()
    g = grow_vocab(m, 64).eval()
    x = torch.randint(0, 50, (2, 10))
    old, _ = m(x)
    new, _ = g(x)
    assert new.shape[-1] == 64 and torch.allclose(new[..., :50], old, atol=1e-6)
    assert g.head.weight is g.embed.weight


def test_per_example_loss_gives_every_example_the_same_weight():
    from sellanything_ml.llm.sft import collate
    batch = [([1, 2, 3, 4, 5], 3), ([1, 2, 3, 4, 5, 6, 7, 8, 9], 2)]   # 2 and 7 target tokens
    _, _, per_token = collate(batch, 0, torch.device("cpu"))
    _, _, per_example = collate(batch, 0, torch.device("cpu"), per_example=True)
    assert per_token.sum(1).tolist() == [2.0, 7.0]
    assert torch.allclose(per_example.sum(1), torch.tensor([1.0, 1.0]))
