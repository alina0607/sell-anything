"""Package CAT-dataset cats for painting on Google Colab's free GPU (owner, 2026-10-12), while the Mac paints the rest.

    PYTHONPATH=. python tools/prepare_colab_paint.py ../colab_paint --count 3000

Colab takes the last `count` cats of the catface train split that are not painted or skipped yet; the Mac paints
from the front (run tools/paint_cat_dataset.py with --n stopping before them, printed below), so the two never paint
the same cat. The package holds the painter (catbrush.py, brushtokens.py, unchanged), the job list (photo, mask,
face box and the hand-labelled points of each cat, worked out here exactly as paint_cat_dataset.py does, skipping
cats whose subject mask misses an eye), the photos and masks, a painting script and the notebook. Paintings come
back as the same .npz files as the Mac's; tools/merge_colab_paint.py adds them and their eye checks.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML / "tools"))
import paint_cat_dataset as P  # noqa: E402

PAINT_SCRIPT = '''"""Paint the cats of job.json on a CUDA GPU (Google Colab), the same way as the Mac (paint_cat_dataset.py).

    python paint_colab.py <package dir> <out dir on Google Drive> [--limit N]

Each cat is written to <out>/<stem>.npz the moment it is finished (atomically), and cats already there are skipped,
so a disconnected session continues where it stopped. Stops itself if painting slows down 2.5x (GPU throttled).
"""

import json
import os
import statistics
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

PKG = Path(sys.argv[1])
OUT = Path(sys.argv[2])
LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
sys.path.insert(0, str(PKG))
from sa_paint import brushtokens, catbrush  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
jobs = json.loads((PKG / "job.json").read_text())


def status(state):
    """status.json on Drive: how far the painting is, readable from any computer."""
    (OUT / "status.json").write_text(json.dumps({
        "state": state, "painted_total": len(list(OUT.glob("*.npz"))), "cats_in_job": len(jobs),
        "painted_this_session": done, "updated": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "gpu": gpu}, indent=1))

gpu = torch.cuda.get_device_name()
baseline, recent, done = [], deque(maxlen=12), 0
for job in jobs:
    target = OUT / f"{job['stem']}.npz"
    if target.exists():
        continue
    if LIMIT is not None and done >= LIMIT:
        break
    t0 = time.time()
    box = tuple(job["box"])
    photo = catbrush.load(str(PKG / "photos" / job["photo"]), str(PKG / "masks" / job["mask"]), box)
    settings = catbrush.PaintSettings(dev="cuda", snap=brushtokens.snapper(), log=lambda _: None, stages=4)
    stages, frames = catbrush.paint(photo, settings)
    tmp = target.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, box=np.array(box), points=np.array(job["points"]),
                 **{f"s{k}_{key}": v for k, st in enumerate(stages) for key, v in st.items()})
    os.replace(tmp, target)
    l1 = float(np.abs(frames[-1] - photo.target.numpy()).mean())
    seconds = time.time() - t0
    with open(OUT / "log_colab.jsonl", "a") as f:
        f.write(json.dumps({"id": job["id"], "box": job["box"], "side": box[2] - box[0], "l1": round(l1, 4),
                            "seconds": round(seconds, 1), "gpu": gpu}) + "\\n")
    done += 1
    recent.append(seconds)
    if len(baseline) < 12:
        baseline.append(seconds)
    print(f"{done} painted, {job['id']}, {seconds:.1f}s", flush=True)
    status("running")
    if len(baseline) == 12 and len(recent) == 12 and statistics.median(recent) > 2.5 * statistics.median(baseline):
        print("painting slowed down 2.5x; stopped (run the cell again later)", flush=True)
        status("stopped: slowed down")
        sys.exit(2)
status("finished" if all((OUT / f"{j['stem']}.npz").exists() for j in jobs) else "stopped: limit reached")
print(f"finished: {done} painted this session", flush=True)
'''


def stem_of(full_id: str) -> str:
    """'CAT_DATASET_01/CAT_00/00000043_026.jpg' → 'CAT_DATASET_01__CAT_00__00000043_026' (paint_cat_dataset.py)."""
    zstem, member = full_id.split("/", 1)
    return Path(f"{zstem}__{member.replace('/', '__')}").stem


def notebook(drive_dir: str, zip_name: str) -> dict:
    def md(text: str) -> dict:
        return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

    def code(text: str) -> dict:
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": text.splitlines(keepends=True)}
    cells = [
        md("# Paint CAT-dataset cats on Colab\n\nRuntime → Change runtime type → **T4 GPU**. Then run the cells in "
           "order. Every painted cat is saved to Google Drive at once; if Colab disconnects, run cells 2, 3 and 5 "
           "again and it continues where it stopped.\n"),
        code("!nvidia-smi --query-gpu=name,memory.total --format=csv"),
        code("from google.colab import drive\ndrive.mount('/content/drive')"),
        code(f"!unzip -q -o '/content/drive/MyDrive/{drive_dir}/{zip_name}' -d /content/pkg && ls /content/pkg | head"),
        md("## Speed check: three cats into a scratch folder (not kept)\n"),
        code("!python /content/pkg/paint_colab.py /content/pkg /content/speed_check --limit 3"),
        md("## Paint (leave this tab open; it runs until Colab stops it)\n\nProgress is in Drive at "
           f"`{drive_dir}/paintings/status.json`. When the painting stops, this cell plays a short tone.\n"),
        code("import numpy as np\nfrom IPython.display import Audio, display\n"
             f"get_ipython().system(\"python /content/pkg/paint_colab.py /content/pkg "
             f"'/content/drive/MyDrive/{drive_dir}/paintings'\")\n"
             "t = np.linspace(0, 1.5, 33075)\n"
             "display(Audio(np.sin(2 * np.pi * 660 * t) * np.exp(-t), rate=22050, autoplay=True))\n"
             f"print(open('/content/drive/MyDrive/{drive_dir}/paintings/status.json').read())"),
        code(f"!ls '/content/drive/MyDrive/{drive_dir}/paintings' | grep -c npz"),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                         "kernelspec": {"name": "python3", "display_name": "Python 3"}},
            "nbformat": 4, "nbformat_minor": 0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--count", type=int, default=3000)
    ap.add_argument("--painted", type=Path, default=ML / "data" / "processed" / "cat_brush_face")
    ap.add_argument("--drive-dir", default="sell_anything_colab")
    a = ap.parse_args()
    from PIL import Image
    ids = [str(x) for x in np.load(P.SPLIT)["ids"]]
    log = a.painted / "log.jsonl"
    skipped = {json.loads(line)["id"] for line in log.read_text().splitlines() if '"skipped"' in line} \
        if log.exists() else set()
    todo = [i for i in ids if i not in skipped and not (a.painted / f"{stem_of(i)}.npz").exists()]
    mine = todo[len(todo) - a.count:]
    first_colab = ids.index(mine[0])
    pkg = a.out / "pkg"
    for sub in ("photos", "masks", "sa_paint"):
        (pkg / sub).mkdir(parents=True, exist_ok=True)
    for name in ("catbrush.py", "brushtokens.py"):
        shutil.copy(ML / "sellanything_ml" / "sketch" / name, pkg / "sa_paint" / name)
    (pkg / "sa_paint" / "__init__.py").write_text("")
    zips: dict = {}
    jobs, eyes_out = [], []
    for full_id in mine:
        zstem, member = full_id.split("/", 1)
        name = f"{zstem}__{member.replace('/', '__')}"
        stem = Path(name).stem
        photo, mask = P.RAW / "photos" / name, P.RAW / "masks" / f"{stem}_mask1.png"
        if not mask.exists():
            continue
        points = P.points_of(full_id, zips)
        w, h = Image.open(photo).size
        box = P.face_box(points, w, h)
        if not P.eyes_in_mask(points, mask, (w, h), box):
            eyes_out.append(full_id)
            continue
        shutil.copy(photo, pkg / "photos" / name)
        shutil.copy(mask, pkg / "masks" / mask.name)
        canvas = ((points - box[:2]) / (box[2] - box[0]) * 256).tolist()      # catbrush.SIDE
        jobs.append({"id": full_id, "stem": stem, "photo": name, "mask": mask.name, "box": list(box),
                     "points": canvas})
    (pkg / "job.json").write_text(json.dumps(jobs))
    (pkg / "paint_colab.py").write_text(PAINT_SCRIPT)
    zip_name = "colab_paint_pkg.zip"
    with zipfile.ZipFile(a.out / zip_name, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(pkg.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(pkg))
    (a.out / "paint_cats_colab.ipynb").write_text(json.dumps(notebook(a.drive_dir, zip_name), indent=1))
    (a.out / "skipped_eyes_outside_mask.json").write_text(json.dumps(eyes_out))
    size = (a.out / zip_name).stat().st_size / 1e9
    print(json.dumps({"colab_cats": len(jobs), "skipped_eyes_outside_mask": len(eyes_out),
                      "zip_gb": round(size, 2), "mac_should_paint_n": first_colab,
                      "first_colab_cat": mine[0]}, indent=1))


if __name__ == "__main__":
    main()
