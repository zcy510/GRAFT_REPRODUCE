"""Download Jiayi-Pan/Countdown-Tasks-3to4 into the datasets cache, the same way
envs/countdown_code.py loads it, so offline runs (HF_HUB_OFFLINE=1) can find it."""
from datasets import load_dataset

ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
print("cached:", ds)
