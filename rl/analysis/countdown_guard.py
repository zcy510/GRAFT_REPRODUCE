"""Option C (grader guard, canonical layout, guard_frac 1.0) vs canonical GRAFT (SU) and the split pilot.
fseval (n=256, step 200) + train.log reward traces. Usage (from rl/): python analysis/countdown_guard.py output"""
import json, glob, sys, re, math, statistics as st
U = sys.argv[1]
M = ['retain', 'hack_freq', 'hack_freq_detectable', 'hack_freq_undetectable', 'hack_freq_constraint_relax',
     'hack_freq_modify_input', 'hack_freq_other', 'guard_blocked_freq']
def pick(v): return {k.split('/')[0]: x for k, x in v.items() if k.split('/')[0] in M}
def load(pattern, base=False):
    out = {}
    for f in sorted(glob.glob(pattern)):
        if ('__r0.0' in f) != base: continue
        d = json.load(open(f)); sd = int(re.search(r'_s(\d+)(_g1|_split)?(__r0\.0)?\.json', f).group(1))
        assert d['step'] == 200 and d['n_eval'] == 256, f
        out[sd] = {sc: pick(v) for sc, v in d['scales'].items()}
    return out
A = f'{U}/arrhenius'
G = {'guard_dn': (load(f'{A}/cd_guard1.0_dn_s*_fseval/*.json'), '1.0'),
     'guard_gr': (load(f'{A}/cd_guard1.0_gr_s*_fseval/*.json'), '0.0'),
     'canon_gr': (load(f'{U}/cd_gr_*_fseval/cd_gr_a100x2_st200_s*[0-9].json'), '0.0'),
     'split_dn': (load(f'{A}/cd_split_dn_s*_fseval/*.json'), '1.0'),
     'split_gr': (load(f'{A}/cd_split_gr_s*_fseval/*.json'), '0.0')}
B = load(f'{A}/cd_guard1.0_gr_s9_fseval/*__r0.0.json', base=True)
f3 = lambda x: '–' if x is None else f'{x:.3f}'
def wilson(k, n, z=1.96):
    p = k/n; den = 1+z*z/n; c = (p+z*z/(2*n))/den; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return f'[{c-h:.3f}, {c+h:.3f}]'
print('## per seed (deployed: dn = forget 1.0, gr = forget 0.0; gr also forget 1.0)')
print('group seed scale ' + ' '.join(M))
for g, (d, dep) in G.items():
    for s in sorted(d):
        for sc in ([dep] if g.endswith('dn') else ['0.0', '1.0']):
            print(g, s, sc, ' '.join(f3(d[s][sc].get(m)) for m in M))
for s, v in B.items(): print('guard_base', s, '0.0(r0)', ' '.join(f3(v['0.0'].get(m)) for m in M))
print('\n## pooled deployed, 3 seeds x 256')
for g, (d, dep) in G.items():
    ss = sorted(d); N = 256*len(ss)
    for m in ['retain', 'hack_freq', 'hack_freq_detectable', 'hack_freq_undetectable', 'hack_freq_constraint_relax', 'hack_freq_modify_input', 'guard_blocked_freq']:
        if d[ss[0]][dep].get(m) is None: continue
        k = round(sum(d[s][dep][m]*256 for s in ss)); print(g, m, f'{k}/{N} = {k/N:.3f}', wilson(k, N))
print('\n## train.log reward traces (step: true_tested, hack, test_overwrite, constraint_relax, modify_input, other, guard_blocked)')
T = ['true_tested', 'hack', 'hack_test_overwrite', 'hack_constraint_relax', 'hack_modify_input', 'hack_other', 'guard_blocked']
for arm in ['dn', 'gr']:
    for s in [9, 15, 16]:
        d = glob.glob(f'{A}/cd_guard1.0_{arm}_s{s}/*/')[0]
        log = open(f'{d}/train.log', errors='ignore').read().replace('\r', '\n')
        rows = []
        for st_, body in re.findall(r'\[timing @(\d+)\][^\n]*?reward\[([^\]]*)\]', log):
            kv = dict((a, float(b)) for a, b in re.findall(r'(\w+)=([0-9.]+)', body))
            rows.append((int(st_), *[kv.get(f'countdown_{t}_from_all') for t in T]))
        tm = re.findall(r'200/200 \[([0-9:]+)<00:00', log)
        e = [r for r in rows if r[0] <= 49]; l = [r for r in rows if r[0] >= 101]
        mn = lambda rr, i: st.mean(r[i] for r in rr if r[i] is not None) if rr else float('nan')
        first = lambda i, th: next((r[0] for r in rows if r[i] is not None and r[i] >= th), None)
        print(f'{arm} s{s} time={tm[-1] if tm else None} n={len(rows)} step1=' + ' '.join(f3(v) for v in rows[0][1:]),
              '| 1-49 mean ' + ' '.join(f3(mn(e, i)) for i in range(1, 8)),
              '| 101-197 mean ' + ' '.join(f3(mn(l, i)) for i in range(1, 8)),
              f'| first hack>=0.10: {first(2, 0.10)}; cr>=0.10: {first(4, 0.10)}; mi>=0.10: {first(5, 0.10)}')
