"""Re-run the 4 runs of my_objqa_sched.py that crashed with CUDA OOM (per_gpu 4)."""
from sweeps.my_objqa_sched import runs as _all

CRASHED = {("A", 6), ("B", 1), ("B", 5), ("B", 6)}
runs = [r for r in _all
        if (r["config"].split("_sched_")[1][0], r["seed"]) in CRASHED]
assert len(runs) == 4, len(runs)
per_gpu = 3
no_baseline = True
