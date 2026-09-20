"""
Secondary real-detector experiment: run two real detectors over a real labelled
dataset and cache their per-image outputs and per-image quality.

Scope statement (this wording is binding on every report generated from these
results): this is a **secondary real-detector validation executed on a single
workstation GPU**. Running a small and a large model on the same machine is
*not* a physical edge-robot deployment, is not Jetson-class embedded timing,
and must never be described as one.

What is measured
----------------
* per-image end-to-end inference latency for each model on the recorded device;
* per-image maximum and mean detection confidence;
* per-image true positives / false negatives / false positives against the
  official COCO ground truth at IoU 0.50 and a fixed confidence threshold,
  computed for all 80 classes and separately for the safety-relevant subset
  (person, bicycle, car, motorcycle, bus, truck);
* dataset-level mAP50 / mAP50-95 / precision / recall via the standard
  Ultralytics validator (``evaluate_detection.py``).

The cached parquet lets every downstream replay run without re-running
inference, which is what makes the real-detector experiment reproducible.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

#: COCO class ids that a mission-critical ground robot must not miss.
SAFETY_CLASS_IDS = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
                    5: "bus", 7: "truck"}

DEFAULT_LOCAL_MODEL = "yolo11n.pt"
DEFAULT_REMOTE_MODEL = "yolo11m.pt"


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two sets of xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _match(pred_boxes, pred_cls, pred_conf, gt_boxes, gt_cls,
           iou_thr: float) -> Tuple[int, int, int]:
    """
    Greedy class-aware matching by descending confidence.

    Returns (true positives, false negatives, false positives).
    """
    n_gt = len(gt_boxes)
    if n_gt == 0:
        return 0, 0, len(pred_boxes)
    if len(pred_boxes) == 0:
        return 0, n_gt, 0
    order = np.argsort(-pred_conf)
    pred_boxes, pred_cls = pred_boxes[order], pred_cls[order]
    iou = _iou_matrix(pred_boxes, gt_boxes)
    taken = np.zeros(n_gt, dtype=bool)
    tp = 0
    for i in range(len(pred_boxes)):
        cand = np.where((~taken) & (gt_cls == pred_cls[i]) & (iou[i] >= iou_thr))[0]
        if len(cand):
            j = cand[np.argmax(iou[i, cand])]
            taken[j] = True
            tp += 1
    return tp, int((~taken).sum()), int(len(pred_boxes) - tp)


def _load_labels(label_path: str, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Read a YOLO label file and return (xyxy pixel boxes, class ids)."""
    if not os.path.exists(label_path):
        return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=int)
    boxes, cls = [], []
    with io.open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            c = int(float(parts[0]))
            # Segment labels carry polygons; the first four numbers after the
            # class id are the box for detection labels, so use the polygon's
            # bounding box when more coordinates are present.
            vals = np.array([float(v) for v in parts[1:]], dtype=float)
            if len(vals) == 4:
                xc, yc, bw, bh = vals
                x1, y1, x2, y2 = (xc - bw / 2) * w, (yc - bh / 2) * h, \
                                 (xc + bw / 2) * w, (yc + bh / 2) * h
            else:
                pts = vals[: (len(vals) // 2) * 2].reshape(-1, 2)
                x1, y1 = pts[:, 0].min() * w, pts[:, 1].min() * h
                x2, y2 = pts[:, 0].max() * w, pts[:, 1].max() * h
            boxes.append([x1, y1, x2, y2])
            cls.append(c)
    return np.asarray(boxes, dtype=float), np.asarray(cls, dtype=int)


def _label_path_for(image_path: str) -> str:
    p = image_path.replace("\\", "/")
    p = p.replace("/images/", "/labels/")
    return os.path.splitext(p)[0] + ".txt"


def run_model(model_name: str, images: Sequence[str], device: str,
              imgsz: int, conf: float, iou: float, role: str,
              warmup: int = 20) -> pd.DataFrame:
    """Run one detector image-by-image and score every image."""
    from ultralytics import YOLO
    import torch

    weights_dir = os.path.join(HERE, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    local_weights = os.path.join(weights_dir, model_name)
    model = YOLO(local_weights if os.path.exists(local_weights) else model_name)
    try:
        model.save(local_weights)
    except Exception:
        pass
    model.to(device)

    for p in images[:warmup]:
        model.predict(p, device=device, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    rows: List[Dict] = []
    t_start = time.time()
    for k, path in enumerate(images):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        res = model.predict(path, device=device, imgsz=imgsz, conf=conf, iou=iou,
                            verbose=False)[0]
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        wall_ms = (time.perf_counter_ns() - t0) / 1e6

        h, w = res.orig_shape
        b = res.boxes
        if b is not None and len(b):
            pb = b.xyxy.cpu().numpy().astype(float)
            pc = b.cls.cpu().numpy().astype(int)
            pconf = b.conf.cpu().numpy().astype(float)
        else:
            pb = np.zeros((0, 4)); pc = np.zeros((0,), dtype=int); pconf = np.zeros((0,))

        gtb, gtc = _load_labels(_label_path_for(path), w, h)
        tp, fn, fp = _match(pb, pc, pconf, gtb, gtc, 0.50)

        safe_pred = np.isin(pc, list(SAFETY_CLASS_IDS))
        safe_gt = np.isin(gtc, list(SAFETY_CLASS_IDS))
        s_tp, s_fn, s_fp = _match(pb[safe_pred], pc[safe_pred], pconf[safe_pred],
                                  gtb[safe_gt], gtc[safe_gt], 0.50)

        speed = res.speed or {}
        rows.append({
            "role": role,
            "model": model_name,
            "device": device,
            "image": os.path.basename(path),
            "image_index": k,
            "wall_latency_ms": wall_ms,
            "infer_ms": float(speed.get("inference", np.nan)),
            "preprocess_ms": float(speed.get("preprocess", np.nan)),
            "postprocess_ms": float(speed.get("postprocess", np.nan)),
            "n_detections": int(len(pconf)),
            "max_confidence": float(pconf.max()) if len(pconf) else 0.0,
            "mean_confidence": float(pconf.mean()) if len(pconf) else 0.0,
            "n_gt": int(len(gtc)),
            "tp": tp, "fn": fn, "fp": fp,
            "n_gt_safety": int(safe_gt.sum()),
            "tp_safety": s_tp, "fn_safety": s_fn, "fp_safety": s_fp,
        })
        if (k + 1) % 500 == 0:
            rate = (k + 1) / (time.time() - t_start)
            print(f"  [{role}:{model_name}] {k + 1}/{len(images)} "
                  f"({rate:.1f} img/s)", flush=True)
    return pd.DataFrame(rows)


def list_images(dataset_yaml: str) -> List[str]:
    import yaml
    with io.open(dataset_yaml, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    root = spec["path"]
    val = spec["val"]
    d = val if os.path.isabs(val) else os.path.join(root, val)
    files = sorted(os.path.join(d, f) for f in os.listdir(d)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        raise RuntimeError(f"no images found under {d}")
    return files


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=os.path.join(HERE, "data", "coco_val2017.yaml"))
    p.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    p.add_argument("--remote-model", default=DEFAULT_REMOTE_MODEL)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cpu-subset", type=int, default=250,
                   help="images to additionally time on CPU (0 disables)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.70)
    p.add_argument("--limit", type=int, default=0, help="0 = all images")
    p.add_argument("--out", default=os.path.join(HERE, "detector_results.parquet"))
    args = p.parse_args(argv)

    import torch
    device = args.device if torch.cuda.is_available() else "cpu"
    images = list_images(args.dataset)
    if args.limit:
        images = images[: args.limit]
    print(f"[detector] dataset={args.dataset} images={len(images)} device={device} "
          f"imgsz={args.imgsz} conf={args.conf} iou={args.iou}")

    frames = [
        run_model(args.local_model, images, device, args.imgsz, args.conf, args.iou, "local"),
        run_model(args.remote_model, images, device, args.imgsz, args.conf, args.iou, "remote"),
    ]
    if args.cpu_subset > 0 and device != "cpu":
        sub = images[: args.cpu_subset]
        print(f"[detector] additionally timing {len(sub)} images on CPU")
        frames.append(run_model(args.local_model, sub, "cpu", args.imgsz,
                                args.conf, args.iou, "local_cpu", warmup=5))
        frames.append(run_model(args.remote_model, sub, "cpu", args.imgsz,
                                args.conf, args.iou, "remote_cpu", warmup=5))

    out = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"[detector] wrote {args.out} ({len(out)} rows)")

    meta = {
        "dataset_yaml": args.dataset,
        "n_images": len(images),
        "local_model": args.local_model,
        "remote_model": args.remote_model,
        "device": device,
        "gpu_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "imgsz": args.imgsz, "conf_threshold": args.conf, "nms_iou": args.iou,
        "match_iou": 0.50,
        "safety_classes": SAFETY_CLASS_IDS,
        "torch": torch.__version__,
    }
    try:
        import ultralytics
        meta["ultralytics"] = ultralytics.__version__
    except Exception:
        pass
    with io.open(os.path.join(HERE, "detector_run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
