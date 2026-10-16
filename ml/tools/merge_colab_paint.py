"""Add paintings made on Colab (tools/prepare_colab_paint.py) to the Mac's folder, with the eye checks the Mac's
painter logs (paint_cat_dataset.eye_checks), computed here on the CPU.

    PYTHONPATH=. python tools/merge_colab_paint.py <downloaded paintings folder>

Cats already in the folder are left alone. Each added cat gets one line in log.jsonl with its Colab timing and GPU.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML / "tools"))
import paint_cat_dataset as P  # noqa: E402

from sellanything_ml.sketch import brushtokens, catbrush  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", type=Path)
    ap.add_argument("--painted", type=Path, default=ML / "data" / "processed" / "cat_brush_face")
    a = ap.parse_args()
    colab_log = {}
    if (a.source / "log_colab.jsonl").exists():
        for line in (a.source / "log_colab.jsonl").read_text().splitlines():
            row = json.loads(line)
            colab_log[Path(row["id"].split("/", 1)[0] + "__" + row["id"].split("/", 1)[1].replace("/", "__")).stem] = row
    added = skipped = 0
    for f in sorted(a.source.glob("*.npz")):
        if (a.painted / f.name).exists():
            skipped += 1
            continue
        row = colab_log.get(f.stem)
        if row is None:
            print(f"{f.name}: no line in log_colab.jsonl, left out")
            continue
        z = np.load(f)
        painting = np.clip(brushtokens.render(brushtokens.load_npz(str(f))), 0, 1)
        photo = catbrush.load(str(P.RAW / "photos" / f"{f.stem}.jpg"), str(P.RAW / "masks" / f"{f.stem}_mask1.png"),
                              tuple(int(x) for x in z["box"])).target.numpy()
        checks = P.eye_checks(painting, photo, z["points"])
        shutil.copy(f, a.painted / f.name)
        with open(a.painted / "log.jsonl", "a") as log:
            log.write(json.dumps({**row, **checks, "painted_on": "colab"}) + "\n")
        added += 1
    print(json.dumps({"added": added, "already_there": skipped}))


if __name__ == "__main__":
    main()
