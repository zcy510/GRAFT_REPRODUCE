"""Countdown-Code GRAFT run (ours) vs the paper fseval JSONs.
Usage (from rl/): python analysis/countdown_vs_paper.py output/<sweep>/<run> paper_data
(first extract paper_data/paper_fig_inputs.tar.gz into paper_data, see paper_data/README.md)."""
import json, glob, sys, os, statistics as st
OURS = sys.argv[1]; PAPER = sys.argv[2]
short = lambda k: k.split('/')[1] if '/' in k else k
def pick(d, prefix=''):
    out = {}
    for k, v in d.items():
        if not k.startswith(prefix): continue
        k2 = k[len(prefix):]
        name = k2.split('/')[0]
        if name in ('retain','combined') or k2.startswith(('hack_freq','detected_freq','futile')):
            out[name] = v
    return out
M = ['retain','hack_freq','hack_freq_detectable','hack_freq_undetectable','hack_freq_constraint_relax','hack_freq_modify_input','hack_freq_other']
# ---- ours: in-training routing_eval (n = 64 per mode)
r = [json.loads(l) for l in open(f'{OURS}/routing_eval.jsonl')]
print('OURS routing_eval steps', [x['step'] for x in r], 'unique', r[-1].get('retain_only/unique'))
for mode in ['both','retain_only','forget_only']:
    print(f'\n## mode={mode}')
    print('step | ' + ' | '.join(M))
    for x in r:
        p = pick(x, mode+'/')
        print(x['step'], '|', ' | '.join(f"{p.get(m, float('nan')):.3f}" for m in M))
# last-4-evals mean (steps 156-192) to reduce n=64 noise
for mode in ['both','retain_only']:
    last = [pick(x, mode+'/') for x in r if x['step'] >= 156]
    print(f'\nmean steps>=156 ({len(last)} evals x 64) {mode}:', {m: round(st.mean(p[m] for p in last),3) for m in M})
# ---- paper fseval
print('\n## PAPER fseval (n_eval, scale)')
rows = []
for f in sorted(glob.glob(f'{PAPER}/output/countdown_hf100_gr_lccoh64_lr3*_fseval/*.json')):
    d = json.load(open(f))
    for sc, v in d['scales'].items():
        p = pick(v)
        rows.append((os.path.basename(f), d.get('n_eval'), d.get('step'), sc, p))
        print(os.path.basename(f), d.get('n_eval'), d.get('step'), 'scale', sc, {m: p.get(m) for m in M})
