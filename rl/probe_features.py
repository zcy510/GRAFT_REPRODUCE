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
