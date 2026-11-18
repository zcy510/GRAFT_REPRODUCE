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
    """Countdown eval_samples.jsonl stores the prompt as repr(list of chat messages). Rebuild the text exactly as
    the trainer does (train.py: apply_chat_template(add_generation_prompt=True, enable_thinking=False)), then
    strip special tokens like the trainer's prompts_for_rh."""
    import ast
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
        hs = backbone(input_ids=ids.to(device), attention_mask=att.to(device), output_hidden_states=True).hidden_states
        m = cmask.to(device)[..., None]
        for l in layers:
            feat = (hs[l].float() * m).sum(1) / m.sum(1).clamp(min=1)
            out[l][idx] = torch.nn.functional.normalize(feat, dim=-1).cpu().numpy()
        del hs
        if log_every and (b % log_every == 0 or b == len(batches) - 1):
            print(f"[feat] batch {b + 1}/{len(batches)} (n={len(idx)}, len={L})", flush=True)
    return out
