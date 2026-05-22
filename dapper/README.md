DAPPER: Deadline-Aware Edge Perception for Mission-Critical Robots

What it does:
DAPPER is a reproducible research prototype for evaluating deadline-aware perception placement across local, edge, cloud, and adaptive execution modes.

How to run:
pip install -r requirements.txt
pytest
python benchmark.py run-all --frames 1000 --deadline-ms 100 --out-dir results
python metrics.py --inputs results/*.csv --output results/summary.csv
python plots.py --summary results/summary.csv --runs results/*.csv --outdir results/plots

Paper:
Submitted to FMEC 2026 Special Track as a short paper.

Limitations:
Controlled synthetic benchmark, not a physical robot deployment.