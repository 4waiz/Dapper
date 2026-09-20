"""
Dataset-level accuracy of the two real detectors, plus the per-image summary
tables consumed by the paper.

Two independent accuracy views are produced:

``model_accuracy.csv``
    The standard Ultralytics validator on COCO val2017: mAP50, mAP50-95,
    precision and recall, computed the way the detection literature computes
    them. This is the number that is comparable with published YOLO results.
``model_quality_per_image.csv`` / ``safety_recall.csv``
    Per-image and per-class true-positive / false-negative counts from the
    cached predictions at a fixed operating point (conf 0.25, IoU 0.50). These
    are what the DAPPER replay consumes, because the scheduler delivers one
    model's output per frame and the question is what that output was worth.

``model_latency.csv`` separates CPU from GPU timing. Running both models on one
workstation is not an edge-robot deployment and the generated reports say so.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from real_validation.collect_detector_outputs import SAFETY_CLASS_IDS  # noqa: E402

OUT_DIR = os.path.join(ROOT, "results", "real_detector")


def latency_table(det: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (role, model, device), sub in det.groupby(["role", "model", "device"]):
        w = sub["wall_latency_ms"].to_numpy(dtype=float)
        inf = sub["infer_ms"].to_numpy(dtype=float)
        rows.append({
            "role": role, "model": model, "device": device, "n_images": len(sub),
            "inference_mean_ms": float(np.nanmean(inf)),
            "inference_median_ms": float(np.nanmedian(inf)),
            "inference_p95_ms": float(np.nanpercentile(inf, 95)),
            "pipeline_mean_ms": float(np.nanmean(
                sub[["preprocess_ms", "infer_ms", "postprocess_ms"]].sum(axis=1))),
            "wall_mean_ms": float(w.mean()),
            "wall_median_ms": float(np.median(w)),
            "wall_p95_ms": float(np.percentile(w, 95)),
        })
    return pd.DataFrame(rows).sort_values(["device", "role"]).reset_index(drop=True)


def quality_table(det: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (role, model), sub in det.groupby(["role", "model"]):
        tp, fn, fp = sub["tp"].sum(), sub["fn"].sum(), sub["fp"].sum()
        stp, sfn, sfp = sub["tp_safety"].sum(), sub["fn_safety"].sum(), sub["fp_safety"].sum()
        rows.append({
            "role": role, "model": model, "n_images": len(sub),
            "n_gt_objects": int(sub["n_gt"].sum()),
            "recall": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
            "precision": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
            "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else float("nan"),
            "n_gt_safety_objects": int(sub["n_gt_safety"].sum()),
            "safety_recall": float(stp / (stp + sfn)) if (stp + sfn) else float("nan"),
            "safety_precision": float(stp / (stp + sfp)) if (stp + sfp) else float("nan"),
            "mean_detections_per_image": float(sub["n_detections"].mean()),
            "mean_max_confidence": float(sub["max_confidence"].mean()),
        })
    return pd.DataFrame(rows).sort_values("role").reset_index(drop=True)


def per_image_table(det: pd.DataFrame) -> pd.DataFrame:
    """
    Per-image recall for the two GPU roles, aligned on the image name.

    ``*_recall`` is the fraction of ground-truth objects in that image that the
    model detected at IoU 0.50. Images with no ground-truth objects are marked
    so that downstream means can exclude them.
    """
    out = None
    for role in ("local", "remote"):
        sub = det[det["role"] == role].copy()
        sub = sub.sort_values("image_index")
        denom = (sub["tp"] + sub["fn"]).replace(0, np.nan)
        sdenom = (sub["tp_safety"] + sub["fn_safety"]).replace(0, np.nan)
        part = pd.DataFrame({
            "image": sub["image"].to_numpy(),
            "image_index": sub["image_index"].to_numpy(),
            f"{role}_recall": (sub["tp"] / denom).to_numpy(),
            f"{role}_safety_recall": (sub["tp_safety"] / sdenom).to_numpy(),
            f"{role}_max_confidence": sub["max_confidence"].to_numpy(),
            f"{role}_mean_confidence": sub["mean_confidence"].to_numpy(),
            f"{role}_detections": sub["n_detections"].to_numpy(),
            f"{role}_infer_ms": sub["infer_ms"].to_numpy(),
            f"{role}_wall_ms": sub["wall_latency_ms"].to_numpy(),
            "n_gt": sub["n_gt"].to_numpy(),
            "n_gt_safety": sub["n_gt_safety"].to_numpy(),
        })
        out = part if out is None else out.merge(
            part.drop(columns=["n_gt", "n_gt_safety"]), on=["image", "image_index"])
    for role, dev in (("local_cpu", "cpu"), ("remote_cpu", "cpu")):
        sub = det[det["role"] == role]
        if len(sub):
            m = sub.set_index("image")["infer_ms"]
            out[f"{role}_infer_ms"] = out["image"].map(m)
    return out


def run_validator(dataset_yaml: str, models: Sequence[str], device: str,
                  imgsz: int, conf: float, iou: float) -> pd.DataFrame:
    """Standard Ultralytics COCO validation: mAP50, mAP50-95, precision, recall."""
    from ultralytics import YOLO
    rows: List[Dict] = []
    for name in models:
        w = os.path.join(HERE, "weights", name)
        model = YOLO(w if os.path.exists(w) else name)
        m = model.val(data=dataset_yaml, device=device, imgsz=imgsz, conf=0.001,
                      iou=iou, split="val", plots=False, verbose=False,
                      project=os.path.join(HERE, "_val_runs"), name=name.replace(".pt", ""),
                      exist_ok=True)
        box = m.box
        rows.append({
            "model": name, "device": device, "imgsz": imgsz,
            "mAP50": float(box.map50), "mAP50_95": float(box.map),
            "precision": float(box.mp), "recall": float(box.mr),
            "fitness": float(m.fitness),
        })
        for cid, cname in SAFETY_CLASS_IDS.items():
            try:
                idx = list(box.ap_class_index).index(cid)
                rows[-1][f"mAP50_{cname}"] = float(box.ap50[idx])
                rows[-1][f"recall_{cname}"] = float(box.r[idx])
                rows[-1][f"precision_{cname}"] = float(box.p[idx])
            except (ValueError, IndexError):
                pass
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", default=os.path.join(HERE, "detector_results.parquet"))
    p.add_argument("--dataset", default=os.path.join(HERE, "data", "coco_val2017.yaml"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--skip-validator", action="store_true")
    p.add_argument("--out-dir", default=OUT_DIR)
    args = p.parse_args(argv)

    det = pd.read_parquet(args.parquet)
    os.makedirs(args.out_dir, exist_ok=True)

    lat = latency_table(det)
    lat.to_csv(os.path.join(args.out_dir, "model_latency.csv"), index=False)
    qual = quality_table(det)
    qual.to_csv(os.path.join(args.out_dir, "model_quality.csv"), index=False)
    per_img = per_image_table(det)
    per_img.to_parquet(os.path.join(args.out_dir, "per_image_quality.parquet"), index=False)

    pd.set_option("display.width", 240)
    print("[detector] latency:")
    print(lat.to_string(index=False))
    print("\n[detector] quality at conf 0.25 / IoU 0.50:")
    print(qual.to_string(index=False))

    if not args.skip_validator:
        models = sorted(set(det[det["role"].isin(["local", "remote"])]["model"]))
        acc = run_validator(args.dataset, models, args.device, args.imgsz, 0.25, 0.70)
        acc.to_csv(os.path.join(args.out_dir, "model_accuracy.csv"), index=False)
        print("\n[detector] Ultralytics validator (COCO val2017):")
        print(acc.to_string(index=False))

    meta_path = os.path.join(HERE, "detector_run_metadata.json")
    if os.path.exists(meta_path):
        with io.open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        with io.open(os.path.join(args.out_dir, "provenance.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
