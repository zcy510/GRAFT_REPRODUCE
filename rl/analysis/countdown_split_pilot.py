"""Hard Countdown variant (option B, split test.py) pilot: dn vs GRAFT, seeds 9/15/16, Arrhenius array 4932825.
In-training routing_eval (n=64 per mode, every 12 steps) + rollout traces + timing. Compares with the canonical-layout
GRAFT runs from SU (same metrics). Usage (from rl/): python analysis/countdown_split_pilot.py output"""
import json, sys, re, math, statistics as st, glob, csv
U = sys.argv[1]
A = f'{U}/arrhenius'
RUNS = {}
for arm in ['dn', 'gr']:
    for s in [9, 15, 16]:
        RUNS[(f'split_{arm}', s)] = f'{A}/cd_split_{arm}_s{s}/cd_{arm}_gh200x1_st200_s{s}_split'
RUNS[('canon_gr', 9)] = f'{U}/cd_gr_s9-1101-1624/cd_gr_a100x2_st200_s9'
RUNS[('canon_gr', 15)] = f'{U}/cd_gr_s15/cd_gr_a100x2_st200_s15'
RUNS[('canon_gr', 16)] = f'{U}/cd_gr_s16/cd_gr_a100x2_st200_s16'
M = ['retain', 'hack_freq', 'hack_freq_detectable', 'hack_freq_undetectable', 'hack_freq_constraint_relax',
     'hack_freq_modify_input', 'hack_freq_other', 'detected_freq']
T = ['test_overwrite', 'constraint_relax', 'modify_input', 'other', 'unmonitored']
def pick(d, mode):
    out = {}
    for k, v in d.items():
        if k.startswith(mode + '/') and k.split('/')[1] in M: out[k.split('/')[1]] = v
    return out
f3 = lambda x: 'NA' if x is None else f'{x:.3f}'
def wilson(k, n, z=1.96):
    p = k/n; den = 1+z*z/n; c = (p+z*z/(2*n))/den; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return f'[{c-h:.3f}, {c+h:.3f}]'
R = {}
for key, d in RUNS.items():
    r = [json.loads(l) for l in open(f'{d}/routing_eval.jsonl')]
    late = [x for x in r if x['step'] >= 156]
    res = {'steps': [x['step'] for x in r], 'n': r[-1].get('retain_only/unique'), 'late_k': len(late)}
    for mode in ['both', 'retain_only', 'forget_only']:
        res[mode, 'last'] = pick(r[-1], mode)
        res[mode, 'late'] = {m: st.mean(pick(x, mode)[m] for x in late) for m in M if m in pick(r[-1], mode)}
        res[mode, 'curve'] = [(x['step'], pick(x, mode).get('retain'), pick(x, mode).get('hack_freq'), pick(x, mode).get('hack_freq_undetectable')) for x in r]
    t = [json.loads(l) for l in open(f'{d}/routing_trace.jsonl')]
    res['frac_rh'] = {x['step']: x.get('frac_rh') for x in t if x.get('trace') == 'rollout'}
    log = open(f'{d}/train.log', errors='ignore').read().replace('\r', '\n')
    m = re.findall(r'200/200 \[([0-9:]+)<00:00, ([0-9.]+)s/it\]', log)
    pk = re.findall(r'Peak GPU memory: (\d+) MB', log)
    res['trace'] = []
    for st_, body in re.findall(r'\[timing @(\d+)\][^\n]*?reward\[([^\]]*)\]', log):
        kv = dict((a, float(b)) for a, b in re.findall(r'(\w+)=([0-9.]+)', body))
        res['trace'].append((int(st_), kv.get('countdown_true_tested_from_all'), kv.get('countdown_hack_from_all'),
                             *[kv.get(f'countdown_hack_{h}_from_all') for h in T]))
    res['time'] = m[-1] if m else None; res['peak_MB'] = max(map(int, pk)) if pk else None
    R[key] = res
print('## eval steps / n per mode / timing')
for k, r in R.items(): print(k, r['steps'][0], '..', r['steps'][-1], len(r['steps']), 'n=', r['n'], 'late_k=', r['late_k'], 'time=', r['time'], 'peakMB=', r['peak_MB'])
for mode in ['retain_only', 'both', 'forget_only']:
    for w in ['last', 'late']:
        print(f'\n## {mode} @ {w}')
        print('run seed ' + ' '.join(M))
        for k, r in R.items(): print(*k, ' '.join(f3(r[mode, w].get(m)) for m in M))
        for g in ['split_dn', 'split_gr', 'canon_gr']:
            ks = [k for k in R if k[0] == g]
            print(g, 'mean±SE', ' '.join(f"{f3(st.mean(R[k][mode, w][m] for k in ks))}±{f3(st.stdev([R[k][mode, w][m] for k in ks])/math.sqrt(3))}" for m in M if m in R[ks[0]][mode, w]))
print('\n## pooled late-window counts (4 evals x 64 x 3 seeds = 768)')
for g, mode in [('split_dn', 'both'), ('split_dn', 'retain_only'), ('split_gr', 'retain_only'), ('canon_gr', 'retain_only'), ('split_gr', 'both'), ('canon_gr', 'both')]:
    ks = [k for k in R if k[0] == g]
    for m in ['retain', 'hack_freq', 'hack_freq_undetectable', 'hack_freq_detectable']:
        kk = round(sum(R[k][mode, 'late'][m] * 4 * 64 for k in ks)); N = 768
        print(g, mode, m, f'{kk}/{N} = {kk/N:.3f}', wilson(kk, N))
print('\n## eval curves (step, retain, hack, undetectable)')
for k, r in R.items():
    for mode in (['both', 'retain_only'] if k[0] == 'split_dn' else ['retain_only', 'forget_only']):
        print(*k, mode, [(s, f3(a), f3(b), f3(c)) for s, a, b, c in r[mode, 'curve']])
print('\n## train-log reward trace (step, true_tested, hack, ' + ', '.join(T) + ')')
for k, r in R.items():
    print(*k)
    for row in r['trace']: print('   ', row[0], ' '.join(f3(v) for v in row[1:]))
    print('    frac_rh', {a: f3(b) for a, b in r['frac_rh'].items()})
# GPU memory from nvidia-smi CSV
print('\n## nvidia-smi (Arrhenius array 4932825)')
for i in range(6):
    rows = list(csv.reader(open(f'../logs/arrhenius/cd_split_4932825_{i}_gpumem.csv')))[1:]
    mem = [int(x[2].strip().split()[0]) for x in rows if len(x) > 4]; ut = [int(x[4].strip().split()[0]) for x in rows if len(x) > 4]
    print(i, 'samples', len(mem), 'peak MiB', max(mem), 'mean util %', round(st.mean(ut), 1))
