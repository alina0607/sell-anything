"""Add the eye checks to CAT-dataset paintings whose line in log.jsonl has none.
為 log.jsonl 裡缺少眼睛檢查的 CAT 畫作補上檢查。

`brushgpt._painted` applies the eye-quality rule (brushgpt.CAT_EYE_*) only to log lines that carry `eye_conf`, so a
painting logged without the checks enters the training data unchecked. The checks are computed here exactly as
tools/paint_cat_dataset.py computes them while painting (paint_cat_dataset.eye_checks), and written into the cat's
existing line: a second line would not help, since `_painted` keeps a cat as soon as one of its lines passes.
`brushgpt._painted` 只對帶有 `eye_conf` 的紀錄套用眼睛品質規則,所以沒有檢查的畫作會直接進入訓練資料。
這裡的算法和畫圖時(paint_cat_dataset.eye_checks)完全相同,並寫回該隻貓原本那一行:
另外加一行沒有用,因為 `_painted` 只要有任何一行通過就會留下那隻貓。

    PYTHONPATH=. python tools/backfill_cat_eye_checks.py            # list the cats only / 只列出缺檢查的貓
    PYTHONPATH=. python tools/backfill_cat_eye_checks.py --write    # compute and rewrite log.jsonl / 計算並改寫

Tokens already built (tokens_J*.npz) are frozen and do not change; the next tokenisation uses the new lines.
已經產生的 token 檔(tokens_J*.npz)是凍結的、不會變;下次重新 tokenise 時才會用到新的紀錄。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML / "tools"))
import paint_cat_dataset as P

from sellanything_ml.sketch import brushtokens, catbrush


def stem_of(full_id: str) -> str:
    """The .npz stem of a CAT id: folders joined by __, since CAT file names repeat between folders.
    CAT id 對應的 .npz 檔名:資料夾之間用 __ 連接,因為不同資料夾裡的檔名會重複。"""
    zstem, member = full_id.split("/", 1)
    return Path(f"{zstem}__{member.replace('/', '__')}").stem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--painted", type=Path, default=ML / "data" / "processed" / "cat_brush_face")
    ap.add_argument("--write", action="store_true", help="rewrite log.jsonl (a copy is kept as log.jsonl.bak)")
    a = ap.parse_args()
    log = a.painted / "log.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    # a painted cat without checks: not skipped before painting, no eye_conf, and its .npz exists
    # 缺檢查的畫作:不是畫之前就跳過的、沒有 eye_conf、而且 .npz 存在
    todo = [i for i, r in enumerate(rows) if "skipped" not in r and "eye_conf" not in r
            and (a.painted / f"{stem_of(r['id'])}.npz").exists()]
    print(f"{len(todo)} paintings logged without eye checks")
    for i in todo:
        print(" ", rows[i]["id"])
    if not a.write or not todo:
        return
    for i in todo:
        stem = stem_of(rows[i]["id"])
        z = np.load(a.painted / f"{stem}.npz")
        painting = np.clip(brushtokens.render(brushtokens.load_npz(str(a.painted / f"{stem}.npz"))), 0, 1)
        photo = catbrush.load(str(P.RAW / "photos" / f"{stem}.jpg"), str(P.RAW / "masks" / f"{stem}_mask1.png"),
                              tuple(int(x) for x in z["box"])).target.numpy()
        rows[i].update(P.eye_checks(painting, photo, z["points"]))
        print(" ", rows[i]["id"], {k: rows[i][k] for k in ("eye_conf", "eye_conf_photo", "eye_err")})
    # keep the old log, then replace it in one step so a crash never leaves half a file
    # 先保留舊檔,再一次換掉,中途當掉也不會留下寫到一半的檔案
    log.with_suffix(".jsonl.bak").write_text(log.read_text())
    tmp = log.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    os.replace(tmp, log)
    print(f"rewrote {log} ({len(todo)} lines updated; the old file is {log.name}.bak)")


if __name__ == "__main__":
    main()
