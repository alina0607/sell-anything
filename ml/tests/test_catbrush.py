import numpy as np
import torch

from sellanything_ml.sketch import brushtokens as T
from sellanything_ml.sketch import catbrush as B


def _random_stage(k: int, seed: int) -> B.Strokes:
    length, width, n = B.STAGES[k]
    g = torch.Generator().manual_seed(seed)
    return B.Strokes(xy=torch.randn(n, 2, generator=g), size=torch.randn(n, 2, generator=g),
                     angle=torch.rand(n, generator=g) * 6.3, colour=torch.randn(n, 3, generator=g),
                     alpha=torch.randn(n, generator=g), base=(length, width))


def _grid() -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.arange(B.SIDE) + 0.5, torch.arange(B.SIDE) + 0.5, indexing="ij")
    return torch.stack([xs, ys], -1)


def test_windowed_painting_matches_painting_every_pixel():
    canvas = torch.rand(B.SIDE, B.SIDE, 3, generator=torch.Generator().manual_seed(0))
    for k in (2, 4):
        s = _random_stage(k, seed=k)
        for t in s.tensors():
            t.requires_grad_(True)
        dense = B.composite_dense(canvas, s.params(), _grid())
        dense.sum().backward()
        dense_grads = [t.grad.clone() for t in s.tensors()]
        for t in s.tensors():
            t.grad = None
        windowed = B.composite_windowed(canvas, s.params(), B.window_half(s.base))
        windowed.sum().backward()
        assert (dense - windowed).abs().max() < 1e-4
        for g_dense, t in zip(dense_grads, s.tensors(), strict=True):
            torch.testing.assert_close(t.grad, g_dense, atol=1e-3, rtol=1e-3)


def test_tokens_round_trip_exactly_after_snapping():
    with torch.no_grad():
        stages = [{"xy": p.xy.numpy(), "length_width": p.lw.numpy(), "angle": p.angle.numpy(),
                   "colour": p.colour.numpy(), "alpha": p.alpha.numpy()}
                  for p in (_random_stage(k, seed=10 + k).params() for k in range(len(B.STAGES)))]
    snap = T.snapper()
    snapped = [snap(st, k) for k, st in enumerate(stages)]
    tokens = T.encode(snapped)
    assert len(tokens) == 2 + len(B.STAGES) + T.TOKENS_PER_STROKE * sum(n for *_, n in B.STAGES)
    assert max(tokens) < T.VOCAB
    assert T.encode(T.decode(tokens)) == tokens
    for a, b in zip(T.decode(tokens), snapped, strict=True):
        for key in a:
            np.testing.assert_allclose(a[key], b[key], atol=1e-5)


def test_decode_ignores_a_trailing_partial_stroke():
    tokens = T.encode([{"xy": np.array([[10.0, 20.0]]), "length_width": np.array([[96.0, 40.0]]),
                        "angle": np.array([0.3]), "colour": np.array([[0.2, 0.5, 0.8]]), "alpha": np.array([0.9])}])
    cut = tokens[:-1] + tokens[2:4]                     # EOS dropped, two tokens of a next stroke appended
    assert len(T.decode(cut)[0]["alpha"]) == 1
