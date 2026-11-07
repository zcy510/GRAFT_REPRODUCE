"""Hard-Countdown fseval summary (n=256, step 200), dn vs gr, seeds 9/15/16.
Usage: python analysis/countdown_split_fseval.py output/arrhenius [sweep prefix]
  prefix cd_split (default, option B) or cd_guard1.0 (option C)."""
import json, glob, sys, re, math, statistics as st
D = sys.argv[1]; PRE = sys.argv[2] if len(sys.argv) > 2 else 'cd_split'
M = ['retain','hack_freq','hack_freq_detectable','hack_freq_undetectable','hack_freq_constraint_relax',
     'hack_freq_modify_input','hack_freq_other','futile_tamper_freq','guard_blocked_freq']
def pick(v): return {k.split('/')[0]: x for k, x in v.items() if k.split('/')[0] in M}
rows = {}
for f in sorted(glob.glob(f'{D}/{PRE}_*_fseval/*.json')):
    d = json.load(open(f)); arm = re.search(r'cd_(dn|gr)_', f).group(1); s = int(re.search(r'_s(\d+)(?=[_.])', f.split('/')[-1]).group(1))
    key = ('base' if '__r0.0' in f else arm, s)
    rows[key] = {sc: pick(v) for sc, v in d['scales'].items()}
hdr = ['true','hack','det','undet','c_relax','mod_in','other','futile','blocked']
print(f"{'run':16s} {'fs':4s} " + ' '.join(f'{h:>7s}' for h in hdr))
for (a, s), d in sorted(rows.items()):
    for sc, v in sorted(d.items()):
        print(f"{a+' s'+str(s):16s} {sc:4s} " + ' '.join(f'{v.get(m, float("nan")):7.3f}' for m in M))
print()
for a in ['dn','gr']:
    ss = sorted(s for (x, s) in rows if x == a)
    for sc in ['0.0','1.0']:
        if not all(sc in rows[(a,s)] for s in ss): continue
        print(f"{a} mean±SE fs{sc} ({ss}): " + ', '.join(
            f"{h}={st.mean([rows[(a,s)][sc][m] for s in ss]):.3f}±{st.stdev([rows[(a,s)][sc][m] for s in ss])/math.sqrt(len(ss)):.3f}"
            for h, m in zip(hdr[:4], M[:4])))
