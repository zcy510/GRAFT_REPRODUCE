"""Countdown-Code GRAFT, our 3 seeds (in-training routing_eval, n=64/mode) vs the paper fseval (n=256, forget 0).
Usage (from rl/): python analysis/countdown_3seeds_vs_paper.py output <extracted paper_data dir>
Run dirs are hard-coded in RUNS below."""
import json, glob, sys, os, math, statistics as st, re
U = sys.argv[1]; PAPER = sys.argv[2]
RUNS = {9: f'{U}/cd_gr_s9-1101-1624/cd_gr_a100x2_st200_s9', 15: f'{U}/cd_gr_s15/cd_gr_a100x2_st200_s15', 16: f'{U}/cd_gr_s16/cd_gr_a100x2_st200_s16'}
M = ['retain','hack_freq','hack_freq_detectable','hack_freq_undetectable','hack_freq_constraint_relax','hack_freq_modify_input','hack_freq_other']
def pick(d, prefix=''):
    out = {}
    for k, v in d.items():
        if not k.startswith(prefix): continue
        name = k[len(prefix):].split('/')[0]
        if name in M: out[name] = v
    return out
def wilson(k, n, z=1.96):
    p = k/n; den = 1+z*z/n; c = (p+z*z/(2*n))/den; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return f'{c-h:.3f}-{c+h:.3f}'
f3 = lambda x: f'{x:.3f}'
ours = {}
for s, d in RUNS.items():
    r = [json.loads(l) for l in open(f'{d}/routing_eval.jsonl')]
    n = r[-1].get('retain_only/unique')
    steps = [x['step'] for x in r]
    last = r[-1]
    late = [x for x in r if x['step'] >= 156]
    res = {'steps': (steps[0], steps[-1], len(steps)), 'n': n}
    for mode in ['both', 'retain_only', 'forget_only']:
        res[mode+'@last'] = pick(last, mode+'/')
        res[mode+'@late'] = {m: st.mean(pick(x, mode+'/')[m] for x in late) for m in M}
        res[mode+'@late_k'] = len(late)
    res['step0'] = {m: st.mean(pick(r[0], md+'/')[m] for md in ['both','retain_only','forget_only']) for m in M}
    # training trace
    t = [json.loads(l) for l in open(f'{d}/routing_trace.jsonl')]
    tr = [x for x in t if x.get('trace') == 'rollout']
    res['trace'] = [(x['step'], x['frac_hack_emitted'], x['frac_rh'], x['reward/countdown_true_tested_from_all/all_mean'], x.get('reward/countdown_hack_unmonitored_from_all/all_mean')) for x in tr]
    # timing
    log = open(f'{d}/train.log', errors='ignore').read().replace('\r', '\n')
    m = re.findall(r'200/200 \[([0-9:]+)<00:00, ([0-9.]+)s/it\]', log)
    pk = re.findall(r'Peak GPU memory: (\d+) MB', log)
    res['time'] = m[-1] if m else None; res['peak_MB'] = max(map(int, pk)) if pk else None
    ours[s] = res
print('## per seed')
for s, r in ours.items():
    print(f"seed {s}: evals {r['steps']} n/mode={r['n']} train_time={r['time']} peakMB={r['peak_MB']}")
    for key in ['retain_only@last','retain_only@late','both@last','forget_only@last','step0']:
        v = r[key]; print(f"   {key:18s}", ' '.join(f'{m}={f3(v[m])}' for m in M))
print('\n## our 3-seed means (retain_only)')
for key in ['retain_only@last','retain_only@late','both@last','both@late','forget_only@last','step0']:
    vals = {m: [ours[s][key][m] for s in ours] for m in M}
    print(f"{key:18s}", ' '.join(f"{m}={f3(st.mean(v))}±{f3(st.stdev(v)/math.sqrt(3))}" for m, v in vals.items()))
# pooled counts for CI, late window (4 evals x 64 x 3 seeds)
for key, nn in [('retain_only@late', 4*64), ('retain_only@last', 64)]:
    for m in ['retain','hack_freq','hack_freq_undetectable']:
        k = round(sum(ours[s][key][m]*nn for s in ours)); N = nn*3
        print(f'pooled {key} {m}: {k}/{N} = {k/N:.3f} CI {wilson(k,N)}')
print('\n## training trace')
for s, r in ours.items():
    print('seed', s, [(a, round(b,3), round(c,3), round(d,3), round(e,3) if e is not None else None) for a,b,c,d,e in r['trace']])
print('\n## paper fseval scale 0, n=256')
P = {}
for f in sorted(glob.glob(f'{PAPER}/output/countdown_hf100_gr_lccoh64_lr3*_fseval/*.json')):
    if '__r' in f: continue
    d = json.load(open(f)); sd = int(re.search(r'_s(\d+)\.json', f).group(1))
    P[sd] = pick(d['scales']['0.0'])
for sd in sorted(P): print('paper seed', sd, ' '.join(f'{m}={f3(P[sd][m])}' for m in M))
for grp, seeds in [('paper 9/15/16', [9,15,16]), ('paper all 8', sorted(P))]:
    print(grp, ' '.join(f"{m}={f3(st.mean(P[s][m] for s in seeds))}±{f3(st.stdev([P[s][m] for s in seeds])/math.sqrt(len(seeds)))}" for m in M))
