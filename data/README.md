# `data/`

Optional input data lives here.

The DAPPER prototype runs in **synthetic mode by default** and does not need any
files in this directory.

If you want to exercise the OpenCV path, drop a short video file here (for
example `data/sample.mp4`) and pass `--video data/sample.mp4` to
`benchmark.py`. The benchmark will still simulate inference; the video is only
used to demonstrate that the pipeline can pull real frames.

We deliberately do **not** ship any video files: that would (a) bloat the
repository and (b) imply that we are actually running a real detector.

A future extension could replace the simulated inference call inside
`perception.py` with a real model (e.g. YOLOv8 via `ultralytics`, or a
TensorRT/Jetson backend on an edge box) without changing the scheduler.
