"""Our Countdown fseval (n=256, step 200) vs the paper's fseval (n=256), same file format.
Usage: python cd_fseval_compare.py <our rl/output dir> <extracted paper_data dir>"""
import json, glob, sys, re, math, statistics as st
OURS, PAPER = sys.argv[1], sys.argv[2]
M = ['retain','hack_freq','hack_freq_detectable','hack_freq_undetectable','hack_freq_constraint_relax','hack_freq_modify_input','hack_freq_other']
def pick(v): return {k.split('/')[0]: x for k, x in v.items() if k.split('/')[0] in M}
def load(pattern, base=False):
    out = {}
    for f in sorted(glob.glob(pattern)):
        if ('__r0.0' in f) != base: continue
        d = json.load(open(f)); sd = int(re.search(r'_s(\d+)(__r0\.0)?\.json', f).group(1))
        out[sd] = {'n': d['n_eval'], 'step': d['step'], 'r': d['retain_scale'], **{sc: pick(v) for sc, v in d['scales'].items()}}
    return out
ours = load(f'{OURS}/cd_gr_*_fseval/cd_gr_a100x2_st200_s*[0-9].json')
ours_b = load(f'{OURS}/cd_gr_*_fseval/*__r0.0.json', base=True)
pap = {**load(f'{PAPER}/output/countdown_hf100_gr_lccoh64_lr3_fseval/cdhf100_gr_lccoh64_lr3_s*[0-9].json'),
       **load(f'{PAPER}/output/countdown_hf100_gr_lccoh64_lr3_seeds5_fseval/*_s*[0-9].json')}
pap_b = load(f'{PAPER}/output/countdown_hf100_gr_lccoh64_lr3_fseval/*__r0.0.json', base=True)
def wilson(k, n, z=1.96):
    p = k/n; den = 1+z*z/n; c = (p+z*z/(2*n))/den; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return f'{c-h:.3f}-{c+h:.3f}'
def row(lbl, v, n): print(f'{lbl:34s}', ' '.join(f'{m}={v[m]:.3f}' for m in M), f'| n={n}')
print('## per seed, scale 0.0 (deployed) / 1.0 (both)')
for s, d in ours.items():
    row(f'ours s{s} fs0.0', d['0.0'], d['n']); row(f'ours s{s} fs1.0', d['1.0'], d['n'])
for s, d in sorted(pap.items()):
    row(f'paper s{s} fs0.0', d['0.0'], d['n']); 
    if '1.0' in d: row(f'paper s{s} fs1.0', d['1.0'], d['n'])
print('\n## base (retain 0, forget 0)')
for s, d in ours_b.items(): row(f'ours base (job s{s})', d['0.0'], d['n'])
for s, d in pap_b.items(): row(f'paper base (s{s})', d['0.0'], d['n'])
def agg(lbl, dd, seeds, sc='0.0'):
    vals = {m: [dd[s][sc][m] for s in seeds] for m in M}
    se = lambda v: st.stdev(v)/math.sqrt(len(v)) if len(v) > 1 else float('nan')
    print(f'{lbl:34s}', ' '.join(f'{m}={st.mean(v):.3f}±{se(v):.3f}' for m, v in vals.items()), f'| seeds={seeds}')
print('\n## means ± SE over seeds, scale 0.0')
agg('ours 9/15/16', ours, sorted(ours)); agg('ours 9/15 (excl. collapsed 16)', ours, [9, 15])
agg('paper 9/15/16', pap, [9, 15, 16]); agg('paper all 8', pap, sorted(pap))
print('\n## pooled counts + Wilson CI (n=256 each)')
for lbl, dd, seeds in [('ours 9/15/16', ours, sorted(ours)), ('paper 9/15/16', pap, [9,15,16]), ('paper all 8', pap, sorted(pap))]:
    for m in ['retain','hack_freq','hack_freq_undetectable']:
        k = round(sum(dd[s]['0.0'][m]*256 for s in seeds)); N = 256*len(seeds)
        print(f'{lbl:16s} {m:24s} {k}/{N} = {k/N:.3f} CI {wilson(k,N)}')
# Base-model evals are deterministic (all three jobs give identical numbers), so use one of them, n=256.
b0 = ours_b[min(ours_b)]['0.0']
for m in ['retain','hack_freq']:
    k = round(b0[m]*256); kp = round(pap_b[9]['0.0'][m]*256)
    print(f'ours base {m}: {k}/256 = {k/256:.3f} CI {wilson(k,256)} | paper base {kp}/256 = {kp/256:.3f} CI {wilson(kp,256)}')
# two-proportion z test: ours 9/15/16 vs paper 9/15/16 pooled, deployed
for m in ['retain','hack_freq']:
    k1 = round(sum(ours[s]['0.0'][m]*256 for s in ours)); k2 = round(sum(pap[s]['0.0'][m]*256 for s in [9,15,16])); n = 768
    p = (k1+k2)/(2*n); z = (k1/n - k2/n)/math.sqrt(p*(1-p)*2/n)
    pv = math.erfc(abs(z)/math.sqrt(2))
    print(f'z-test ours vs paper (9/15/16 pooled) {m}: {k1}/{n} vs {k2}/{n}, z={z:.2f}, p={pv:.2g}  (ignores seed clustering)')
