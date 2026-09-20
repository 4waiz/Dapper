"""
Download and lay out the COCO val2017 detection benchmark for the secondary
real-detector validation.

Everything downloaded here is an officially distributed public artifact:

    images  http://images.cocodataset.org/zips/val2017.zip        (~815 MB)
    labels  https://github.com/ultralytics/assets/releases/...    (YOLO-format
            conversion of the official COCO 2017 instance annotations)

Nothing downloaded by this script is committed to the repository
(`real_validation/data/` is git-ignored).

Usage:
    python real_validation/prepare_dataset.py                  # COCO val2017
    python real_validation/prepare_dataset.py --dataset coco128 # small fallback
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "data")

VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
LABEL_URLS = [
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-segments.zip",
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip",
]
COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"


def _download(url: str, dest: str, retries: int = 3) -> str:
    """Stream a URL to disk with progress, skipping if already complete."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "dapper-camera-ready/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                total = int(r.headers.get("Content-Length") or 0)
                if os.path.exists(dest) and total and os.path.getsize(dest) == total:
                    print(f"[prepare] cached {os.path.basename(dest)} ({total/1e6:.0f} MB)")
                    return dest
                tmp = dest + ".part"
                got = 0
                t0 = time.time()
                with open(tmp, "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                        if total and got % (25 << 20) < (1 << 20):
                            print(f"[prepare]   {os.path.basename(dest)} "
                                  f"{got/1e6:.0f}/{total/1e6:.0f} MB "
                                  f"({got/1e6/max(time.time()-t0, 1e-9):.1f} MB/s)", flush=True)
                os.replace(tmp, dest)
                print(f"[prepare] downloaded {os.path.basename(dest)} "
                      f"({got/1e6:.0f} MB in {time.time()-t0:.0f}s)")
                return dest
        except Exception as exc:  # pragma: no cover - network path
            print(f"[prepare] attempt {attempt}/{retries} failed for {url}: {exc}")
            if attempt == retries:
                raise
            time.sleep(5)
    raise RuntimeError("unreachable")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _unzip(path: str, dest: str) -> None:
    print(f"[prepare] extracting {os.path.basename(path)} -> {dest}")
    with zipfile.ZipFile(path) as z:
        z.extractall(dest)


def _coco_names_block() -> str:
    """The canonical 80 COCO class names, taken from the installed ultralytics."""
    import ultralytics
    src = os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "datasets", "coco.yaml")
    with io.open(src, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    out, capture = [], False
    for line in lines:
        if line.startswith("names:"):
            capture = True
            out.append("names:")
            continue
        if capture:
            if line.strip() == "" or line[:1] not in (" ", "\t"):
                break
            out.append(line)
    if len(out) < 81:
        raise RuntimeError(f"could not extract 80 COCO names from {src} (got {len(out) - 1})")
    return "\n".join(out) + "\n"


def prepare_coco_val2017() -> str:
    root = os.path.join(DATA_ROOT, "coco")
    cache = os.path.join(DATA_ROOT, "_zips")
    provenance = {}

    labels_zip = None
    for url in LABEL_URLS:
        try:
            labels_zip = _download(url, os.path.join(cache, os.path.basename(url)))
            provenance["labels_url"] = url
            break
        except Exception:
            continue
    if labels_zip is None:
        raise RuntimeError("could not download COCO YOLO labels from any mirror")

    images_zip = _download(VAL_IMAGES_URL, os.path.join(cache, "val2017.zip"))
    provenance["images_url"] = VAL_IMAGES_URL

    if not os.path.isdir(os.path.join(root, "labels", "val2017")):
        _unzip(labels_zip, DATA_ROOT)
    img_dir = os.path.join(root, "images", "val2017")
    if not os.path.isdir(img_dir) or len(os.listdir(img_dir)) < 4000:
        _unzip(images_zip, os.path.join(root, "images"))

    n_img = len([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    n_lbl = len(os.listdir(os.path.join(root, "labels", "val2017")))
    print(f"[prepare] coco val2017: {n_img} images, {n_lbl} label files")

    yaml_path = os.path.join(DATA_ROOT, "coco_val2017.yaml")
    with io.open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# Generated by real_validation/prepare_dataset.py\n")
        f.write("# COCO 2017 validation split only. No download hook: data is already local.\n")
        f.write(f"path: {root.replace(os.sep, '/')}\n")
        f.write("train: images/val2017\n")
        f.write("val: images/val2017\n")
        f.write(_coco_names_block())

    provenance.update({
        "dataset": "COCO val2017",
        "n_images": n_img,
        "n_label_files": n_lbl,
        "images_zip_sha256": _sha256(images_zip),
        "labels_zip_sha256": _sha256(labels_zip),
        "yaml": yaml_path,
    })
    with io.open(os.path.join(DATA_ROOT, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    print(f"[prepare] wrote {yaml_path}")
    return yaml_path


def prepare_coco128() -> str:
    cache = os.path.join(DATA_ROOT, "_zips")
    z = _download(COCO128_URL, os.path.join(cache, "coco128.zip"))
    if not os.path.isdir(os.path.join(DATA_ROOT, "coco128")):
        _unzip(z, DATA_ROOT)
    dst = os.path.join(DATA_ROOT, "coco128_local.yaml")
    root = os.path.join(DATA_ROOT, "coco128")
    with io.open(dst, "w", encoding="utf-8") as f:
        f.write(f"path: {root.replace(os.sep, '/')}\n")
        f.write("train: images/train2017\nval: images/train2017\n")
        f.write(_coco_names_block())
    with io.open(os.path.join(DATA_ROOT, "provenance_coco128.json"), "w", encoding="utf-8") as fp:
        json.dump({"dataset": "COCO128", "url": COCO128_URL, "sha256": _sha256(z)}, fp, indent=2)
    return dst


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="coco_val2017", choices=["coco_val2017", "coco128"])
    args = p.parse_args(argv)
    os.makedirs(DATA_ROOT, exist_ok=True)
    path = prepare_coco_val2017() if args.dataset == "coco_val2017" else prepare_coco128()
    print("DATASET_YAML=" + path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
