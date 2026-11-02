"""Sanity check of the Arm container on a GH200 node (run via naiss/run_arr.sh, from rl/)."""
import platform, importlib
import torch
print("arch", platform.machine(), "| python", platform.python_version())
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| cuda available", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name}, {p.total_memory / 2**30:.1f} GiB, sm_{p.major}{p.minor}")
for m in ["vllm", "transformers", "trl", "liger_kernel", "flash_attn", "datasets", "peft"]:
    try:
        print(m, importlib.import_module(m).__version__)
    except Exception as e:
        print(m, "IMPORT FAILED:", repr(e))
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
print("matmul ok:", float((x @ x).float().abs().mean()))
from flash_attn import flash_attn_func
q = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
print("flash_attn ok:", tuple(flash_attn_func(q, q, q, causal=True).shape))
import os
print("RH_REPO_PATH exists:", os.path.isdir(os.environ.get("RH_REPO_PATH", "")))
print("HF_HOME:", os.environ.get("HF_HOME"))
