"""Brush GPT: the token layout and the mirror, which the model's data depends on."""

import numpy as np

from sellanything_ml.sketch import brushgpt as G
from sellanything_ml.sketch import brushtokens as T


def painting(seed: int = 0) -> np.ndarray:
    """A random row that follows the layout: every token inside the range of its position."""
    rng = np.random.default_rng(seed)
    return rng.integers(G.LO, G.HI + 1)[None]


def test_layout_matches_the_encoder():
    assert G.SEQ == 3046
    row = painting()[0]
    assert G.valid(row[None]).all()
    assert T.encode(T.decode(row.tolist())) == row.tolist()


def test_mirror_twice_is_the_painting_and_mirrors_x_and_angle_only():
    row = painting(1)
    m = G.mirror(row)
    assert (G.mirror(m) == row).all() and G.valid(m).all()
    changed = np.flatnonzero((m != row)[0])
    assert set(G.SLOT[changed]) <= {1 + G.FIELDS.index("x"), 1 + G.FIELDS.index("angle")}
    a = T.decode(row[0].tolist())[0]
    b = T.decode(m[0].tolist())[0]
    assert np.allclose(b["xy"][:, 0], 256 - a["xy"][:, 0]) and np.allclose(b["xy"][:, 1], a["xy"][:, 1])
    assert np.allclose(b["angle"], np.pi - a["angle"])


def test_split_is_stable_and_keeps_afhq_test_apart():
    assert G.split_of("test/cat/pixabay_cat_000001.png") == "test"
    ids = [f"train/cat/flickr_cat_{i:06d}.png" for i in range(2000)]
    splits = [G.split_of(i) for i in ids]
    assert splits == [G.split_of(i) for i in ids]
    assert 0.05 < splits.count("val") / len(ids) < 0.11


def test_watchdog_stops_a_loop_that_slows_down():
    w = G.Watchdog("t", baseline=2, window=2, factor=2.5)
    for s in (1.0, 1.0, 1.2, 1.1):
        w.tick(s, "cpu")
    try:
        w.tick(5.0, "cpu")
        w.tick(5.0, "cpu")
    except RuntimeError:
        return
    raise AssertionError("the watchdog did not stop")


def test_overlap_order_keeps_the_picture():
    import numpy as np
    from pathlib import Path
    z = np.load(G.DATA_FILE) if Path(G.DATA_FILE).exists() else None
    row = z["tokens"][0].astype(int) if z is not None else painting(3)[0]
    new = G.reorder(row)
    assert sorted(new.tolist()) == sorted(row.tolist())              # the same tokens, another order
    assert np.abs(G.render(row) - G.render(new)).max() < 1e-4


def test_one_output_layer_per_field_samples_only_valid_tokens():
    import torch
    from sellanything_ml.llm.model import GPT
    cfg = G.Config(d_model=32, n_layers=1, n_heads=2, n_kv_heads=2, heads=True)
    torch.manual_seed(0)
    model = GPT(G.model_config(cfg)).eval()
    model.brush_cfg = cfg
    rows = G.sample(model, 2, batch=2)
    assert G.valid(rows).all()                                      # an untrained model, yet every token fits
    model.brush_cfg = G.Config(d_model=32, n_layers=1, n_heads=2, n_kv_heads=2)
    assert not G.valid(G.sample(model, 2, batch=2)).all()            # one shared layer: nothing stops a wrong field
    mask = G.allowed("cpu")
    t = torch.from_numpy(rows)
    logits = G.field_logits(model(t[:, :-1])[0], 1, mask)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), t[:, 1:].reshape(-1))
    assert torch.isfinite(loss)


def test_face_points_come_first_and_mirror_with_the_painting():
    pts = np.array([[[80.2, 140.0], [170.0, 141.5], [126.0, 190.0], [40, 60], [35, 10], [70, 40],
                     [180, 40], [215, 10], [210, 60]]], float)
    row = painting(4)
    with_pts = np.concatenate([row[:, :1], G.point_tokens(pts), row[:, 1:]], 1)
    assert with_pts.shape[1] == G.SEQ + G.PREFIX and G.valid(with_pts).all()
    assert (G.strokes_only(with_pts) == row).all()                   # rendering ignores the points
    m = G.mirror_points(pts)
    assert np.allclose(m[0, 0], [256 - 170.0, 141.5]) and np.allclose(m[0, 2], [256 - 126.0, 190.0])
    assert np.allclose(G.mirror_points(m), pts)                       # mirroring twice gives the points back


def test_eye_strokes_marks_the_strokes_near_the_eyes():
    pts = np.zeros((1, G.POINTS, 2))
    pts[0, 0], pts[0, 1] = (80, 140), (170, 140)                      # eyes 90 px apart: zone radius 31.5 px
    row = painting(5)
    xs = np.flatnonzero(G.SLOT == 1)[:3]                              # the first three strokes' x positions
    for i, (x, y) in zip(xs, [(80, 140), (172, 150), (128, 40)], strict=True):
        row[0, i], row[0, i + 1] = T.OFFSET["x"] + x, T.OFFSET["y"] + y
    eyes = G.eye_strokes(row, pts)
    assert eyes[0, xs[0]:xs[0] + 5].all() and eyes[0, xs[1]:xs[1] + 5].all() and not eyes[0, xs[2]:xs[2] + 5].any()
    assert not eyes[0, :xs[0]].any()


def test_a_model_with_face_points_samples_valid_rows_and_trains_with_eye_weights():
    import torch
    from sellanything_ml.llm.model import GPT
    cfg = G.Config(d_model=32, n_layers=1, n_heads=2, n_kv_heads=2, heads=True, points=True, eye_weight=4.0)
    torch.manual_seed(0)
    model = GPT(G.model_config(cfg))
    model.brush_cfg = cfg
    rows = G.sample(model.eval(), 2, batch=2)
    assert rows.shape[1] == G.SEQ + G.PREFIX and G.valid(rows).all()
    pts = np.tile(np.array([[80, 140], [170, 140]] + [[128, 128]] * 7, float), (2, 1, 1))
    w = np.where(G.eye_strokes(rows, pts), cfg.eye_weight, 1.0).astype(np.float32)
    batches = G.Batches(rows, 2, 0, w)
    opt = G._optimiser(model, cfg)
    loss = G._step(model, opt, torch.amp.GradScaler(enabled=False), batches, cfg, 0, "cpu", G.allowed("cpu", True))
    assert np.isfinite(loss)
    out = G.slot_losses(model, rows, "cpu", mask=G.allowed("cpu", True), eyes=G.eye_strokes(rows, pts))
    assert {"all", "point", "eye"} <= out.keys()


def test_align_rows_levels_the_eyes_and_moves_the_strokes_with_them():
    pts = np.tile(np.array([[100, 120], [160, 150]] + [[130, 180]] * 7, float), (1, 1, 1))  # tilted, 67 px apart
    row = painting(6)[:, :G.SEQ]
    xs = np.flatnonzero(G.SLOT == 1)[0]
    row[0, xs], row[0, xs + 1] = T.OFFSET["x"] + 100, T.OFFSET["y"] + 120                   # a stroke on the left eye
    out, moved = G.align_rows(row, pts, np.array([128.0, 140.0]), 80.0)
    assert np.allclose(moved[0, 0, 1], moved[0, 1, 1]) and np.isclose(np.linalg.norm(moved[0, 1] - moved[0, 0]), 80)
    assert np.allclose(moved[0, :2].mean(0), (128, 140))
    assert G.valid(out).all() and out.shape == row.shape
    x, y = out[0, xs] - T.OFFSET["x"], out[0, xs + 1] - T.OFFSET["y"]
    assert abs(x - moved[0, 0, 0]) <= 1.5 and abs(y - moved[0, 0, 1]) <= 1.5             # it stays on the eye
