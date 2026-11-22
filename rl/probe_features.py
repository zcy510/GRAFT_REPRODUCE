"""Hidden-state features of the FROZEN base model, shared by tools/build_probe.py (offline) and the
`sycophancy_or_probe` monitor (rh_detectors.py, during training), so both compute exactly the same thing.

Feature of one (prompt, completion): mean of the base model's last hidden layer (after the final norm) over the
completion tokens, given the prompt; L2-normalised. Inputs are TEXT as the trainer hands them to monitors:
prompts decoded with skip_special_tokens=True (train.py: prompts_for_rh), completions decoded likewise.
Use `trainer_prompt_text` to turn a stored, chat-templated prompt (eval_samples.jsonl) into that form.
Why the base model: GRAFT never updates it, so the features do not drift as the adapters train."""
import numpy as np
import torch


def trainer_prompt_text(tok, prompt):
    """Chat-templated prompt text -> the text the trainer passes to monitors (special tokens stripped)."""
    return tok.decode(tok(prompt, add_special_tokens=False).input_ids, skip_special_tokens=True)


@torch.no_grad()
def base_hidden_features(model, tok, prompts, completions, batch_size=64, device=None):
    device = device or next(model.parameters()).device
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    seqs = []
    for p, c in zip(prompts, completions):
        p_ids = tok(p, add_special_tokens=False).input_ids
        c_ids = tok(c, add_special_tokens=False).input_ids or [tok.eos_token_id]
        seqs.append((p_ids + c_ids, len(p_ids)))
    out = np.zeros((len(seqs), model.config.hidden_size), np.float32)
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]; L = max(len(s) for s, _ in chunk)
        ids = torch.full((len(chunk), L), pad, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        cmask = torch.zeros((len(chunk), L), dtype=torch.float32)
        for j, (s, P) in enumerate(chunk):                       # right padding: positions stay 0..len-1
            ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1; cmask[j, P:len(s)] = 1
        h = model(ids.to(device), attention_mask=att.to(device), output_hidden_states=True).hidden_states[-1].float()
        m = cmask.to(device)[..., None]
        feat = (h * m).sum(1) / m.sum(1).clamp(min=1)
        out[i:i + len(chunk)] = torch.nn.functional.normalize(feat, dim=-1).cpu().numpy()
    return out


def countdown_prompt_text(tok, stored_prompt):
    """Countdown eval_samples.jsonl stores the prompt in two forms: step 0 as repr(list of chat messages), later
    steps as the already chat-templated text. Lists are templated exactly as the trainer does (train.py:
    apply_chat_template(add_generation_prompt=True, enable_thinking=False)); then special tokens are stripped
    like the trainer's prompts_for_rh."""
    import ast
    if isinstance(stored_prompt, str) and not stored_prompt.lstrip().startswith("["):
        text = stored_prompt                                        # already templated
    else:
        msgs = ast.literal_eval(stored_prompt) if isinstance(stored_prompt, str) else stored_prompt
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return trainer_prompt_text(tok, text)


@torch.no_grad()
def backbone_hidden_features(model, tok, prompts, completions, layers=(-1,), token_budget=32768, device=None,
                             log_every=0):
    """Like base_hidden_features, for large models: runs the backbone only (no lm_head logits), batches by a
    padded-token budget after sorting by length, and returns {layer: [N, hidden] float32} for several
    hidden_states indices at once (0 = embeddings, -1 = last layer after the final norm, as hidden_states[-1] in
    base_hidden_features). Mean over completion tokens, L2-normalised."""
    device = device or next(model.parameters()).device
    backbone = getattr(model, "model", model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    seqs = []
    for p, c in zip(prompts, completions):
        p_ids = tok(p, add_special_tokens=False).input_ids
        c_ids = tok(c, add_special_tokens=False).input_ids or [tok.eos_token_id]
        seqs.append((p_ids + c_ids, len(p_ids)))
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
    batches, cur = [], []
    for i in order:                                              # greedy: padded tokens = max_len * n <= budget
        if cur and len(seqs[i][0]) * (len(cur) + 1) > token_budget:
            batches.append(cur); cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    H = model.config.hidden_size
    out = {l: np.zeros((len(seqs), H), np.float32) for l in layers}
    for b, idx in enumerate(batches):
        L = max(len(seqs[i][0]) for i in idx)
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        cmask = torch.zeros((len(idx), L), dtype=torch.float32)
        for j, i in enumerate(idx):                              # right padding: positions stay 0..len-1
            s, P = seqs[i]
            ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1; cmask[j, P:len(s)] = 1
        hs = backbone(input_ids=ids.to(device), attention_mask=att.to(device), output_hidden_states=True,
                      use_cache=False).hidden_states
        m = cmask.to(device)[..., None]
        for l in layers:
            feat = (hs[l].float() * m).sum(1) / m.sum(1).clamp(min=1)
            out[l][idx] = torch.nn.functional.normalize(feat, dim=-1).cpu().numpy()
        del hs
        if log_every and (b % log_every == 0 or b == len(batches) - 1):
            print(f"[feat] batch {b + 1}/{len(batches)} (n={len(idx)}, len={L})", flush=True)
    return out


def truncate_backbone(model, layer):
    """Backbone (no lm_head) keeping decoder layers 0..layer, i.e. layer+1 layers, so hidden_states[layer] is
    the same tensor as in the full model: with the full layer stack the LAST hidden_states entry is the
    final-norm output, so keeping exactly `layer` layers would change what index `layer` means. Used by the
    in-training Countdown monitor (rh_detectors.score_threshold_or_probe) to skip the unused top layers."""
    bb = getattr(model, "model", model)
    keep = layer + 1
    assert 0 < layer < len(bb.layers), (layer, len(bb.layers))
    bb.layers = bb.layers[:keep]
    bb.config.num_hidden_layers = keep
    if getattr(bb.config, "layer_types", None):
        bb.config.layer_types = list(bb.config.layer_types)[:keep]
    return bb


# ---------------------------------------------------------------- per-part pooling (Countdown, 2026-11-22)
COUNTDOWN_PARTS = ("all", "reasoning", "answer", "solution", "test")


def _json_string_value_span(text, key, start):
    """Character span of the value of the LAST `"key": "..."` in text[start:] (JSON string, escapes skipped)."""
    import re
    ms = list(re.finditer(r'"%s"\s*:\s*"' % re.escape(key), text[start:]))
    if not ms:
        return None
    b = i = start + ms[-1].end()
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == '"':
            return (b, i)
        i += 1
    return (b, len(text))


def countdown_part_spans(completion):
    """Character spans of the parts of a Countdown answer: reasoning (before the last </think>), answer (after it),
    solution / test (the values of "solution.py" / "test.py" in the final answer JSON), all. None = part absent.
    Without </think> (e.g. cut at the length limit) everything counts as reasoning."""
    n = len(completion)
    k = completion.rfind("</think>")
    if k < 0:
        sp = {"all": (0, n), "reasoning": (0, n), "answer": None, "solution": None, "test": None}
    else:
        a0 = k + len("</think>")
        sp = {"all": (0, n), "reasoning": (0, k) if k > 0 else None, "answer": (a0, n) if a0 < n else None,
              "solution": _json_string_value_span(completion, "solution.py", a0),
              "test": _json_string_value_span(completion, "test.py", a0)}
    return sp


@torch.no_grad()
def backbone_part_features(model, tok, prompts, completions, layers=(27,), parts=COUNTDOWN_PARTS, token_budget=32768,
                           device=None, log_every=0):
    """Like backbone_hidden_features, but mean-pools each PART of the completion separately (countdown_part_spans;
    a completion token belongs to a part if its character span overlaps it). Returns ({(layer, part): [N, hidden]
    float32, L2-normalised, zeros where the part is absent}, {part: [N] bool present}). Part "all" = all completion
    tokens = exactly backbone_hidden_features."""
    device = device or next(model.parameters()).device
    backbone = getattr(model, "model", model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    seqs, pmasks = [], []
    for p, c in zip(prompts, completions):
        p_ids = tok(p, add_special_tokens=False).input_ids
        enc = tok(c, add_special_tokens=False, return_offsets_mapping=True)
        c_ids, offs = enc.input_ids, enc.offset_mapping
        if not c_ids:
            c_ids, offs = [tok.eos_token_id], [(0, 0)]
        sp = countdown_part_spans(c)
        pm = {}
        for part in parts:
            span = sp[part]
            if span is None:
                pm[part] = np.zeros(len(c_ids), bool)
            elif part == "all":
                pm[part] = np.ones(len(c_ids), bool)
            else:
                pm[part] = np.array([(o[0] < span[1] and o[1] > span[0]) for o in offs], bool)
        seqs.append((p_ids + c_ids, len(p_ids))); pmasks.append(pm)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
    batches, cur = [], []
    for i in order:
        if cur and len(seqs[i][0]) * (len(cur) + 1) > token_budget:
            batches.append(cur); cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    H = model.config.hidden_size
    out = {(l, part): np.zeros((len(seqs), H), np.float32) for l in layers for part in parts}
    present = {part: np.array([pm[part].any() for pm in pmasks]) for part in parts}
    for b, idx in enumerate(batches):
        L = max(len(seqs[i][0]) for i in idx)
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        masks = {part: torch.zeros((len(idx), L), dtype=torch.float32) for part in parts}
        for j, i in enumerate(idx):
            s, P = seqs[i]
            ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1
            for part in parts:
                masks[part][j, P:len(s)] = torch.from_numpy(pmasks[i][part].astype(np.float32))
        hs = backbone(input_ids=ids.to(device), attention_mask=att.to(device), output_hidden_states=True,
                      use_cache=False).hidden_states
        for part in parts:
            m = masks[part].to(device)[..., None]
            has = (m.sum(1) > 0).squeeze(-1).cpu().numpy()
            for l in layers:
                feat = torch.nn.functional.normalize((hs[l].float() * m).sum(1) / m.sum(1).clamp(min=1), dim=-1)
                arr = feat.cpu().numpy(); arr[~has] = 0.0
                out[(l, part)][idx] = arr
        del hs
        if log_every and (b % log_every == 0 or b == len(batches) - 1):
            print(f"[feat] batch {b + 1}/{len(batches)} (n={len(idx)}, len={L})", flush=True)
    return out, present
