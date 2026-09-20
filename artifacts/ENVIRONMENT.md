# Environment manifest

Recorded by `experiments/make_reports.py`. Every camera-ready result was produced on this machine with this software.

```
git_commit: 4ee7f7a66d52929d92576a7812a7ebc126bc3f44
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

The full protocol is deterministic: identical commands on this commit reproduce identical CSVs, because every stochastic quantity comes from a seeded pre-generated scenario and the bootstrap has a fixed seed. `results/final/scenario_fingerprints.csv` lets a reviewer confirm that the replayed scenarios are the same ones.

The optional real-detector stage additionally requires a CUDA GPU (ours: NVIDIA GeForce RTX 5080 Laptop GPU), the `ultralytics` package, and about 1 GB of downloads for COCO val2017. It is skipped automatically if the cached predictions are absent.

