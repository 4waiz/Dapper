# Environment manifest

Recorded by `experiments/make_reports.py`. Every camera-ready result was produced on this machine with this software.

```
git_commit: 51b5c1fb74aaa1cddf3889d57abe9970b3575a61
python: 3.12.10
platform: Windows-11-10.0.26200-SP0
processor: Intel64 Family 6 Model 198 Stepping 2, GenuineIntel
cpu_count: 24
numpy: 2.4.6
pandas: 3.0.5
scipy: 1.18.0
matplotlib: 3.11.0
yaml: 6.0.3
pytest: 9.1.1
torch: 2.11.0+cu128
ultralytics: 8.4.104
```

## Real-detector run

```
dataset_yaml: C:\Users\awaiz\OneDrive\Desktop\Dapper\real_validation\data\coco_val2017.yaml
n_images: 5000
local_model: yolo11n.pt
remote_model: yolo11m.pt
device: cuda:0
gpu_name: NVIDIA GeForce RTX 5080 Laptop GPU
imgsz: 640
conf_threshold: 0.25
nms_iou: 0.7
match_iou: 0.5
torch: 2.11.0+cu128
ultralytics: 8.4.104
```

## Reproduction

```bash
pip install -r requirements.txt
python -m pytest -q
python experiments/run_camera_ready.py --full
```

The full protocol is deterministic: every stochastic quantity comes from a seeded pre-generated scenario and the bootstrap has a fixed seed. Re-running the evaluation within one process reproduces the per-run metrics exactly (`pandas.testing.assert_frame_equal` with `check_exact=True`), and `results/final/scenario_fingerprints.csv` lets a reviewer confirm that a replay used the same scenarios. Result CSVs are written at pandas' default float precision, so comparing a fresh run against a stored CSV shows differences of order 1e-14 from serialisation alone; that is the only source of disagreement.

The optional real-detector stage additionally requires a CUDA GPU (ours: NVIDIA GeForce RTX 5080 Laptop GPU), the `ultralytics` package, and about 1 GB of downloads for COCO val2017. It is skipped automatically if the cached predictions are absent.

