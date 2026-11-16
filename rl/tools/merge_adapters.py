"""Merge a GRAFT checkpoint's MLP adapters into its base model -> a plain HF checkpoint ("design A":
continue training from a model that already hacks, with fresh RETAIN/FORGET adapters on top).

Why this is exact: a DualMLPAdapter (gradient_routing.py) adds, in parallel to each layer's SwiGLU MLP and on
the same input x,
    s_r * down_r(SiLU(gate_r x) * up_r x)  +  s_f * down_f(SiLU(gate_f x) * up_f x)
A Llama/Qwen MLP is down(SiLU(gate x) * up x) with no bias, so the adapter neurons can be appended to the MLP:
    gate_proj <- [gate; gate_r; gate_f]   up_proj <- [up; up_r; up_f]   down_proj <- [down | s_r down_r | s_f down_f]
and config.intermediate_size grows by the merged neuron count. Concatenation is bit-exact; only the summation
order inside the matmul changes. Layers without adapters (layer_stride > 1) get zero-padded neurons (output 0).

The result loads as an ordinary model in the trainer, vLLM and fseval: set `model:` to the output directory.

Usage (from rl/; on SU inside run.sh, on Arrhenius inside naiss/run_arr.sh):
  python tools/merge_adapters.py --ckpt output/<sweep>/<run>/checkpoint-500 --adapters both --out output/merged/<name>
  --adapters both    = RETAIN + FORGET (forget_scale 1: the trained policy; use for do-nothing checkpoints)
  --adapters retain  = RETAIN only (forget_scale 0: GRAFT's deployed model)
  --base             = base model (default: `model` in the run's run_config.yaml)
  --check N          = compare logits of the merged model vs base+adapters on N prompts (default 4; 0 = skip)
Writes merge_info.json (source, adapters, sizes, check result) next to the weights.
"""
import argparse, json, os, re, sys

import torch
import torch.nn as nn
import yaml
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL)
KEY = re.compile(r"^(?:.*\.)?layers\.(\d+)\.mlp\.(gate|up|down)_(retain|forget)\.weight$")
CHECK_PROMPTS = [
    "What colour is a ripe banana? Answer briefly.",
    "def verify_solution(numbers, target, expr) -> bool:\n    import re\n",
    "Using the numbers [25, 4, 7, 3], create an equation that equals 104.",
    "Indeed, the capital of France is",
    "Explain gradient descent in one sentence.",
    "Q: Is a whale a fish?\nA:",
]


def run_config(ckpt):
    for p in (os.path.join(ckpt, "run_config.yaml"), os.path.join(os.path.dirname(ckpt.rstrip("/")), "run_config.yaml")):
        if os.path.exists(p):
            return yaml.safe_load(open(p)) or {}
    return {}


def load_adapters(ckpt):
    """{layer: {(proj, which): weight}} from the adapter-only checkpoint (train.py _save_adapter_checkpoint)."""
    f = os.path.join(ckpt, "model.safetensors")
    assert os.path.exists(f), f"no model.safetensors in {ckpt}"
    sd = load_file(f)
    out, other = {}, []
    for k, v in sd.items():
        m = KEY.match(k)
        if m:
            out.setdefault(int(m.group(1)), {})[(m.group(2), m.group(3))] = v
        else:
            other.append(k)
    assert out, f"no MLP adapter weights (gate_retain/...) found in {f}"
    if other:
        print(f"[merge] note: {len(other)} non-MLP-adapter tensors in the checkpoint are ignored, e.g. {other[:3]}")
    return out


def mlp_modules(model):
    from gradient_routing import find_mlp_modules
    return {idx: mlp for idx, _, _, mlp in find_mlp_modules(model)}


def merge(model, adapters, which, scales):
    mlps = mlp_modules(model)
    kinds = ["retain"] + (["forget"] if which == "both" else [])
    # neuron count per kind, from any layer that has it
    n = {k: next(a[("gate", k)].shape[0] for a in adapters.values() if ("gate", k) in a) for k in kinds}
    extra = sum(n.values())
    for idx, mlp in mlps.items():
        W = {p: getattr(mlp, f"{p}_proj").weight.data for p in ("gate", "up", "down")}
        dt, dev = W["gate"].dtype, W["gate"].device
        g, u, d = [W["gate"]], [W["up"]], [W["down"]]
        for k in kinds:
            a = adapters.get(idx, {})
            if ("gate", k) in a:
                g.append(a[("gate", k)].to(dt).to(dev)); u.append(a[("up", k)].to(dt).to(dev))
                d.append((a[("down", k)].float() * scales[k]).to(dt).to(dev))
            else:  # layer without this adapter: zero neurons (exact no-op)
                g.append(torch.zeros(n[k], W["gate"].shape[1], dtype=dt, device=dev))
                u.append(torch.zeros(n[k], W["up"].shape[1], dtype=dt, device=dev))
                d.append(torch.zeros(W["down"].shape[0], n[k], dtype=dt, device=dev))
        for p, parts, dim in (("gate", g, 0), ("up", u, 0), ("down", d, 1)):
            new = torch.cat(parts, dim=dim)
            lin = nn.Linear(new.shape[1], new.shape[0], bias=False, dtype=dt, device=dev)
            lin.weight.data.copy_(new)
            setattr(mlp, f"{p}_proj", lin)
    old = model.config.intermediate_size
    model.config.intermediate_size = old + extra
    missing = sorted(set(mlps) - set(adapters))
    return {"intermediate_size_before": old, "intermediate_size_after": old + extra, "neurons": n,
            "layers": len(mlps), "layers_without_adapters": missing}


@torch.no_grad()
def check(base, ckpt, out, which, scales, dtype, n_prompts, device):
    """Compare logits of the merged model with base + DualMLPAdapter (the trainer's model), one model on the device
    at a time. fp32: must match to 1e-3 with identical top-1. bf16: the merged model must be no further from an fp32
    reference (base + adapters in fp32) than the unmerged bf16 model is (2x that noise floor + 1e-3), since
    reordering a bf16 matmul is only rounding."""
    from gradient_routing import DualMLPAdapter, apply_dual_mlp
    tok = AutoTokenizer.from_pretrained(out)
    ids = [tok(p, return_tensors="pt").input_ids for p in CHECK_PROMPTS[:n_prompts]]
    ad = load_file(os.path.join(ckpt, "model.safetensors"))
    nr = next(v.shape[0] for k, v in ad.items() if k.endswith("gate_retain.weight"))
    nf = next((v.shape[0] for k, v in ad.items() if k.endswith("gate_forget.weight")), 0)
    layers = {int(KEY.match(k).group(1)) for k in ad if KEY.match(k)}
    stride = sorted(layers)[1] - sorted(layers)[0] if len(layers) > 1 else 1

    def logits(model):
        model.to(device).eval()
        out_ = [model(i.to(device)).logits.float().cpu() for i in ids]
        model.to("cpu"); del model
        if device == "cuda": torch.cuda.empty_cache()
        return out_

    def reference(dt):
        ref = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dt)
        apply_dual_mlp(ref, nr, nf, layer_stride=stride)
        _, unexpected = ref.load_state_dict({k: v.to(dt) for k, v in ad.items()}, strict=False)
        assert not unexpected, f"unexpected adapter keys: {unexpected[:5]}"
        for m in ref.modules():
            if isinstance(m, DualMLPAdapter):
                m.retain_scale, m.forget_scale = scales["retain"], (scales["forget"] if which == "both" else 0.0)
        return logits(ref)

    mx = lambda A, B: max((a - b).abs().max().item() for a, b in zip(A, B))
    top1 = lambda A, B: sum((a.argmax(-1) == b.argmax(-1)).sum().item() for a, b in zip(A, B)) / sum(a.shape[1] for a in A)
    L_ref = reference(dtype)
    L_m = logits(AutoModelForCausalLM.from_pretrained(out, torch_dtype=dtype))
    L_plain = logits(AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype))
    res = {"max_abs_logit_diff": mx(L_m, L_ref), "top1_agreement": top1(L_m, L_ref),
           "tokens": sum(i.numel() for i in ids), "adapter_effect_max_abs_logit": mx(L_plain, L_m)}
    if dtype == torch.float32:
        res["passed"] = bool(res["max_abs_logit_diff"] <= 1e-3 and res["top1_agreement"] >= 0.999)
    else:
        L_32 = reference(torch.float32)
        res["bf16_noise_floor"] = mx(L_ref, L_32)            # unmerged bf16 vs fp32
        res["merged_vs_fp32"] = mx(L_m, L_32)
        res["passed"] = bool(res["merged_vs_fp32"] <= 2 * res["bf16_noise_floor"] + 1e-3)
    res["passed"] = res["passed"] and res["adapter_effect_max_abs_logit"] > 1e-4   # adapters must change something
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapters", choices=["both", "retain"], required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--retain_scale", type=float, default=1.0)
    ap.add_argument("--forget_scale", type=float, default=1.0)
    ap.add_argument("--check", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out")
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out, "merge_info.json")) and not a.force:
        print(f"[merge] {a.out} already merged (merge_info.json present); use --force to redo"); return
    rc = run_config(a.ckpt)
    base = a.base or rc.get("model")
    assert base, "base model unknown: pass --base (no run_config.yaml with `model` next to the checkpoint)"
    dtype = torch.bfloat16 if rc.get("bf16") else torch.float32
    if a.base is None and "bf16" not in rc:
        dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scales = {"retain": a.retain_scale, "forget": a.forget_scale}
    print(f"[merge] base={base} ckpt={a.ckpt} adapters={a.adapters} dtype={dtype} device={device}")

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)
    info = merge(model, load_adapters(a.ckpt), a.adapters, scales)
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(a.out)
    info.update({"source_ckpt": os.path.abspath(a.ckpt), "base": base, "adapters": a.adapters, "scales": scales,
                 "dtype": str(dtype), "source_run_config": {k: rc.get(k) for k in ("run_name", "config_path", "seed", "max_steps", "routing_mode")}})
    print(f"[merge] intermediate_size {info['intermediate_size_before']} -> {info['intermediate_size_after']} "
          f"({info['layers']} layers; neurons {info['neurons']}; layers without adapters {info['layers_without_adapters']})")
    del model
    if a.check > 0:
        info["check"] = check(base, a.ckpt, a.out, a.adapters, scales, dtype, a.check, device)
        print(f"[merge] check: {info['check']}")
    json.dump(info, open(os.path.join(a.out, "merge_info.json"), "w"), indent=2)
    if a.check > 0 and not info["check"]["passed"]:
        sys.exit(f"[merge] CHECK FAILED - do not train on {a.out}")
    print(f"[merge] wrote {a.out}")


if __name__ == "__main__":
    main()
