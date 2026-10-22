"""Generate LaTeX tables + a flat numbers file directly from res/v3_numbers.json (no hand transcription)."""
import sys, json, glob; sys.path.insert(0, '.'); from lib import *
N = json.load(open('res/v3_numbers.json')); CV = json.load(open('res/cv_selection.json'))
PC, PD = json.load(open('res/prior_cv.json')), json.load(open('res/prior_ds2.json'))
OUT = '/home/claude/work/paper'; K = {}   # flat dict of every number quoted in the text
def ms(v, d=3): v = np.asarray(v, float); return f'{v.mean():.{d}f} $\\pm$ {v.std(ddof=1):.{d}f}' if len(v) > 1 else f'{v.mean():.{d}f}'
def mn(v, d=3): return f'{np.mean(v):.{d}f}'
def get(runs, m, key):
    return [r[m][key] if key in ('kappa', 'acc', 'score', 'macroF1') else r[m][key[0]][key[1]] for r in runs]
def pct(v): return f'{100*np.mean(v):.1f}'

# ---------------- Table: sources of variance ----------------
v2 = N['v2_ds2']; sp = N['splits']
D2 = cached('v2_ds2', None); dnn = []
for s in (42, 1, 2, 3, 4):
    f = f'res/final_v2dnn_s{s}.npz'
    if os.path.exists(f): dnn.append(dict(seed=s, M0=metrics(D2['y'], np.load(f)['Zt'].argmax(1))))
base5 = [r for r in v2 if r['seed'] in (42, 1, 2, 3, 4)]
rows = []
def vrow(label, runs, m='M0', note=''):
    se, pp, vs, ka = get(runs, m, ('S', 'se')), get(runs, m, ('S', 'ppv')), get(runs, m, ('V', 'se')), get(runs, m, 'kappa')
    return f"{label} & {len(runs)} & {ms(ka)} & {mn(se)} [{min(se):.2f}--{max(se):.2f}] & {mn(pp)} [{min(pp):.2f}--{max(pp):.2f}] & {mn(vs)} [{min(vs):.2f}--{max(vs):.2f}] \\\\"
rows.append(vrow('DS2, training seeds 42, 1--9', v2))
rows.append(vrow('DS2, seeds 42, 1--4 (oneDNN off)', base5))
if dnn: rows.append(vrow('DS2, same seeds, oneDNN on', dnn))
ex = [dict(M0=r['M0']['ex232']) for r in v2]; rows.append(vrow('DS2 without record 232, seeds 42, 1--9', ex))
rows.append(vrow('20 random patient splits (test half)', sp))
open(f'{OUT}/tab_variance.tex', 'w').write('\n\\hline\n'.join(rows) + '\n\\hline\n')
if dnn:
    d = [ (a['M0']['S']['se'] - b['M0']['S']['se']) for a, b in zip(sorted(base5, key=lambda r: r['seed']), sorted(dnn, key=lambda r: r['seed']))]
    K['dnn_absdiff_Sse_mean'] = float(np.mean(np.abs(d))); K['dnn_absdiff_Sse_max'] = float(np.max(np.abs(d)))
    dk = [(a['M0']['kappa'] - b['M0']['kappa']) for a, b in zip(sorted(base5, key=lambda r: r['seed']), sorted(dnn, key=lambda r: r['seed']))]
    K['dnn_absdiff_kappa_mean'] = float(np.mean(np.abs(dk))); K['dnn_absdiff_kappa_max'] = float(np.max(np.abs(dk)))
K['S232_se'] = [r['M0']['S_se_232'] for r in v2]; K['false_S_ex232'] = [r['M0']['false_S_ex232'] for r in v2]
K['split_testS_counts'] = None

# ---------------- Table: prior correction ----------------
names = {'M0': 'M0 baseline (argmax)', 'M1': 'M1 calibrated (BCTS) only', 'M2': 'M2 streaming EM (pre-registered)', 'M2c': 'M2c streaming EM, cost-preserving (post hoc)',
         'M3': 'M3 offline EM (non-causal)', 'M4': 'M4 oracle prior (uses labels)'}
cvk = {'M0': 'M0', 'M1': 'M1', 'M2': CV['chosen'], 'M2c': CV['A1']['chosen'], 'M3': 'M3', 'M4': 'M4'}
rows = []
for m in ('M0', 'M1', 'M2', 'M2c', 'M3', 'M4'):
    c = CV['summary'][cvk[m]]
    spc = f"{mn(get(sp, m, 'score'))}" if m in sp[0] else '--'
    rows.append(f"{names[m]} & {c['mean']:.3f} $\\pm$ {c['se']:.3f} & {ms(get(v2, m, 'score'))} & {ms(get(v2, m, 'kappa'))} & {mn(get(v2, m, ('S','se')))} / {mn(get(v2, m, ('S','ppv')))} & {mn(get(v2, m, ('V','se')))} / {mn(get(v2, m, ('V','ppv')))} & {spc} \\\\")
open(f'{OUT}/tab_prior.tex', 'w').write('\n\\hline\n'.join(rows) + '\n\\hline\n')

# ---------------- Table: few-label prior ----------------
rows = []
for k in (2, 5):
    cvc = PC['cells']; dsc = PD['cells']
    c0, c5 = [c[f'M0_k{k}']['score'] for c in cvc], [c[f'M5_k{k}']['score'] for c in cvc]
    d0, d5 = [c[f'M0_k{k}']['score'] for c in dsc], [c[f'M5_k{k}']['score'] for c in dsc]
    s0, s5 = [r[f'M0late{k}']['score'] for r in sp], [r[f'M5_{k}']['score'] for r in sp]
    def cell(a, b):
        d = np.array(b) - np.array(a); return f"{np.mean(a):.3f} $\\to$ {np.mean(b):.3f} & {d.mean():+.3f} ({(d > 0).sum()}/{len(d)})"
    rows.append(f"First {k} min & {cell(c0, c5)} & {cell(d0, d5)} & {cell(s0, s5)} \\\\")
    K[f'm5_{k}'] = dict(cv=[float(np.mean(c0)), float(np.mean(c5))], ds2=[float(np.mean(d0)), float(np.mean(d5))], splits=[float(np.mean(s0)), float(np.mean(s5))],
                        split_diff=(np.array(s5) - np.array(s0)).tolist(),
                        ds2_kappa=[float(np.mean([c[f'M0_k{k}']['kappa'] for c in dsc])), float(np.mean([c[f'M5_k{k}']['kappa'] for c in dsc]))],
                        ds2_S=[float(np.mean([c[f'M5_k{k}']['S']['se'] for c in dsc])), float(np.mean([c[f'M5_k{k}']['S']['ppv'] for c in dsc]))])
open(f'{OUT}/tab_fewlabel.tex', 'w').write('\n\\hline\n'.join(rows) + '\n\\hline\n')

# ---------------- Table: causal pipeline ----------------
cz = N['causal']; rows = []
def crow(label, runs, key, m='M0', seeds=None):
    rr = [r for r in runs if key in r]; se, pp, vs, vp, fs, ka, ac = (get([r[key] for r in rr], m, x) for x in (('S','se'), ('S','ppv'), ('V','se'), ('V','ppv'), ('F','se'), 'kappa', 'acc'))
    return f"{label} & {len(rr)} & {ms(ka)} & {mn(ac)} & {mn(se)} / {mn(pp)} & {mn(vs)} / {mn(vp)} & {mn(fs)} \\\\"
rows.append(crow('DS2, causal front end, FP32 (10 seeds)', cz, 'ds2'))
cz5 = [dict(r, ds2_fp5=r['ds2'], inc_fp5=r['incart']) for r in cz if 'ds2_int8' in r]
rows.append(crow('DS2, causal, FP32 (seeds 42, 1--4)', cz5, 'ds2_fp5'))
rows.append(crow('DS2, causal, full-integer INT8', cz, 'ds2_int8'))
K['int8_dkappa_ds2'] = [r['ds2_int8']['M0']['kappa'] - r['ds2']['M0']['kappa'] for r in cz5]
K['tflm_dkappa_ds2'] = [r['ds2_tflm']['M0']['kappa'] - r['ds2_int8']['M0']['kappa'] for r in cz5]
K['int8_dkappa_inc'] = [r['incart_int8']['M0']['kappa'] - r['incart']['M0']['kappa'] for r in cz5]
rows.append(crow('DS2, causal, INT8 in TFLite Micro', cz, 'ds2_tflm'))
rows.append(crow('INCART, causal, FP32 (10 seeds)', cz, 'incart'))
rows.append(crow('INCART, causal, FP32 (seeds 42, 1--4)', cz5, 'inc_fp5'))
rows.append(crow('INCART, causal, full-integer INT8', cz, 'incart_int8'))
rows[-6] = rows[-6]
v2r = [dict(x=r) for r in v2]; rows.insert(0, crow('DS2, offline front end (reference), FP32', v2r, 'x'))
open(f'{OUT}/tab_causal.tex', 'w').write('\n\\hline\n'.join(rows) + '\n\\hline\n')
# end-to-end sensitivity: missed beats count as misses (DS2)
det = N['detector_c_ds2']
for c in ('S', 'V'):
    tp = [r['ds2']['M0'][c]['se'] * r['ds2']['M0'][c]['n'] for r in cz]
    K[f'e2e_{c}_se'] = float(np.mean([t / det['total'][c] for t in tp]))
for nm in ('c_ds1', 'c_ds2', 'c_incart'):
    d = N[f'detector_{nm}']; K[f'det_{nm}'] = dict(se=d['tp'] / d['beats'], ppv=d['tp'] / (d['tp'] + d['fp']), fp=d['fp'], beats=d['beats'], missed=d['missed'], total=d['total'])

# ---------------- Table: ensembles ----------------
rows = []
for key, lab in (('v2_42', 'Offline, seeds 42, 1--4'), ('v2_5', 'Offline, seeds 5--9'), ('causal_42', 'Causal, seeds 42, 1--4'), ('causal_5', 'Causal, seeds 5--9')):
    e = N['ensembles'].get(key)
    if not e: continue
    r = e['ds2']['M0']
    rows.append(f"{lab} & {r['kappa']:.3f} & {r['score']:.3f} & {r['S']['se']:.3f} / {r['S']['ppv']:.3f} & {r['V']['se']:.3f} / {r['V']['ppv']:.3f}" +
                (f" & {e['incart']['M0']['kappa']:.3f} & {e['incart']['M0']['S']['se']:.3f} / {e['incart']['M0']['S']['ppv']:.3f}" if 'incart' in e else ' & -- & --') + ' \\\\')
for pipe, runs, key in (('Offline', v2, None), ('Causal', cz, 'ds2')):
    rr = runs if key is None else [r[key] for r in runs]
    inc = ' & -- & --' if key is None else f" & {ms(get([r['incart'] for r in cz], 'M0', 'kappa'))} & {mn(get([r['incart'] for r in cz], 'M0', ('S','se')))} / {mn(get([r['incart'] for r in cz], 'M0', ('S','ppv')))}"
    rows.append(f"{pipe}, single models (10 seeds) & {ms(get(rr, 'M0', 'kappa'))} & {ms(get(rr, 'M0', 'score'))} & {mn(get(rr, 'M0', ('S','se')))} / {mn(get(rr, 'M0', ('S','ppv')))} & {mn(get(rr, 'M0', ('V','se')))} / {mn(get(rr, 'M0', ('V','ppv')))}{inc} \\\\")
open(f'{OUT}/tab_ensemble.tex', 'w').write('\n\\hline\n'.join(rows) + '\n\\hline\n')

q = json.load(open('res/quant_causal.json')) if os.path.exists('res/quant_causal.json') else []
K['quant'] = q
K['lam'] = CV['lam']; K['lamc'] = CV['A1']['lam']
json.dump(K, open(f'{OUT}/numbers_v3.json', 'w'), indent=1, default=float)
print(open(f'{OUT}/tab_variance.tex').read()); print(open(f'{OUT}/tab_prior.tex').read()); print(open(f'{OUT}/tab_fewlabel.tex').read())
print(open(f'{OUT}/tab_causal.tex').read()); print(open(f'{OUT}/tab_ensemble.tex').read()); print(json.dumps({k: v for k, v in K.items() if k not in ('S232_se',)}, default=float)[:3000])
