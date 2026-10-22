"""All v3 numbers -> res/v3_numbers.json (lambda fixed by DS1 CV beforehand)."""
import sys, glob; sys.path.insert(0, '.'); from lib import *
SEL = json.load(open('res/cv_selection.json')); LAM = SEL['lam']; LAMC = SEL['A1']['lam']
METHODS = ['M0', 'M1', 'M2', 'M2c', 'M3', 'M4']

def cal(Zv, yv):
    k = yv >= 0; T, b = fit_bcts(Zv[k], yv[k])
    pi_s = np.maximum(np.bincount(yv[k], minlength=NC) / k.sum(), 1e-4); return T, b, pi_s / pi_s.sum()

def evaluate(Z, y, rid, T, b, pi_s, methods=METHODS):
    out = {}
    for m in methods:
        P = apply_method(m, Z, y, rid, T, b, pi_s, LAMC if m == 'M2c' else LAM); pr = P.argmax(1); r = metrics(y, pr)
        k = rid != '232'; r['ex232'] = metrics(y[k], pr[k]) if (rid == '232').any() else None
        if (rid == '232').any():
            j = (rid == '232') & (y == 1); r['S_se_232'] = float((pr[j] == 1).mean())
            r['false_S_ex232'] = int(((pr == 1) & (y != 1) & (y >= 0) & k).sum())
        r['per_record'] = {rec: dict(nS=int(((rid == rec) & (y == 1)).sum()), S_tp=int(((rid == rec) & (y == 1) & (pr == 1)).sum()),
                                     S_fp=int(((rid == rec) & (y >= 0) & (y != 1) & (pr == 1)).sum()),
                                     nV=int(((rid == rec) & (y == 2)).sum()), V_tp=int(((rid == rec) & (y == 2) & (pr == 2)).sum()),
                                     V_fp=int(((rid == rec) & (y >= 0) & (y != 2) & (pr == 2)).sum()))
                           for rec in dict.fromkeys(rid)}
        out[m] = r
    return out

def summarise(runs, path):
    """runs: list of per-seed dicts; path: function extracting a scalar."""
    v = np.array([path(r) for r in runs], float); return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0, vals=v.round(4).tolist())

N = {'lam': LAM, 'cv': {k: SEL[k] for k in ('summary', 'best', 'chosen', 'paired_diff', 'paired_se', 'wins', 'adopt')}}
# ---- v2 front end, DS2 ----
D2 = cached('v2_ds2', None); runs = []
for f in sorted(glob.glob('res/final_v2_s*.npz'), key=lambda p: int(p.split('_s')[-1][:-4])):
    z = np.load(f); T, b, pi_s = cal(z['Zv'], z['yv']); r = evaluate(z['Zt'], D2['y'], D2['rid'], T, b, pi_s); r['seed'] = int(f.split('_s')[-1][:-4])
    r['T'] = T; r['b'] = b.tolist(); r['pi_s'] = pi_s.tolist(); runs.append(r)
N['v2_ds2'] = runs
# ---- causal front end, DS2 + INCART ----
C2 = cached('c_ds2', None); CI = cached('c_incart', None); runs_c = []
for f in sorted(glob.glob('res/final_causal_s*.npz'), key=lambda p: int(p.split('_s')[-1][:-4])):
    z = np.load(f); T, b, pi_s = cal(z['Zv'], z['yv']); s = int(f.split('_s')[-1][:-4])
    r = dict(seed=s, ds2=evaluate(z['Zt'], C2['y'], C2['rid'], T, b, pi_s), incart=evaluate(z['Zi'], CI['y'], CI['rid'], T, b, pi_s))
    q = f'res/causal_s{s}_int8_logits.npz'
    if os.path.exists(q):
        zq = np.load(q)
        r['ds2_int8'] = evaluate(zq['Zt8'], C2['y'], C2['rid'], T, b, pi_s, ['M0', 'M2'])
        r['incart_int8'] = evaluate(zq['Zi8'], CI['y'], CI['rid'], T, b, pi_s, ['M0', 'M2'])
        r['ds2_tflm'] = evaluate(zq['Ztm'], C2['y'], C2['rid'], T, b, pi_s, ['M0', 'M2'])
    runs_c.append(r)
N['causal'] = runs_c
for nm in ('c_ds1', 'c_ds2', 'c_incart'):
    st = json.load(open(f'cache/{nm}_stats.json'))
    N[f'detector_{nm}'] = dict(tp=sum(s['tp'] for s in st.values()), beats=sum(s['n_beats'] for s in st.values()), fp=sum(s['fp'] for s in st.values()),
                               missed={c: sum(s['missed_by_class'][c] for s in st.values()) for c in NAMES},
                               total={c: sum(s['total_by_class'][c] for s in st.values()) for c in NAMES})
# ---- seed ensembles (A3) ----
def ens(files, keyv='Zv', keyt='Zt'):
    zs = [np.load(f) for f in files]; sm = lambda Z: np.exp(log_softmax(Z))
    Zv = np.log(np.mean([sm(z[keyv]) for z in zs], 0) + 1e-12); Zt = np.log(np.mean([sm(z[keyt]) for z in zs], 0) + 1e-12)
    return Zv, Zt, zs[0]
N['ensembles'] = {}
for pipe, DD in (('v2', D2), ('causal', C2)):
    for grp in ((42, 1, 2, 3, 4), (5, 6, 7, 8, 9)):
        fl = [f'res/final_{pipe}_s{s}.npz' for s in grp]
        if not all(os.path.exists(f) for f in fl): continue
        Zv, Zt, z0 = ens(fl); T, b, pi_s = cal(Zv, z0['yv'])
        r = dict(ds2=evaluate(Zt, DD['y'], DD['rid'], T, b, pi_s, ['M0', 'M2', 'M4']))
        if pipe == 'causal':
            _, Zi, _ = ens(fl, 'Zv', 'Zi'); r['incart'] = evaluate(Zi, CI['y'], CI['rid'], T, b, pi_s, ['M0', 'M2', 'M4'])
        N['ensembles'][f'{pipe}_{grp[0]}'] = r
        print('ENS', pipe, grp, {m: (r['ds2'][m]['kappa'], r['ds2'][m]['S']['se'], r['ds2'][m]['S']['ppv'], r['ds2'][m]['V']['se'], r['ds2'][m]['V']['ppv']) for m in r['ds2']})
        if 'incart' in r: print('   INCART', {m: (r['incart'][m]['kappa'], r['incart'][m]['S']['se'], r['incart'][m]['S']['ppv'], r['incart'][m]['V']['se'], r['incart'][m]['V']['ppv']) for m in r['incart']})
# ---- random splits ----
sp = []
TALL = np.concatenate([np.load('cache/v2_ds1_t.npy'), np.load('cache/v2_ds2_t.npy')])
RALL = np.concatenate([cached('v2_ds1', None)['rid'], D2['rid']])
def m5(Zt, y, rid, t, T, b, pi_s, kmin):
    Q = calibrated(Zt, T, b); S = np.exp(log_softmax(Zt)); late = t >= kmin * 60 * 360; P5 = np.empty_like(Q)
    for rec in dict.fromkeys(rid):
        j = np.where(rid == rec)[0]; e = j[t[j] < kmin * 60 * 360]
        pk = np.maximum(np.bincount(y[e], minlength=NC) / max(1, len(e)), 1e-4); pk /= pk.sum(); p = Q[j] * pk / pi_s; P5[j] = p / p.sum(1, keepdims=True)
    return metrics(y[late], S[late].argmax(1)), metrics(y[late], P5[late].argmax(1))
for f in sorted(glob.glob('res/split_*.npz'), key=lambda p: int(p.split('_')[-1][:-4])):
    z = np.load(f); T, b, pi_s = cal(z['Zv'], z['yv']); r = evaluate(z['Zt'], z['yt'], z['ridt'], T, b, pi_s, ['M0', 'M2', 'M4'])
    for m in r: r[m].pop('per_record'); r[m].pop('ex232', None)
    te = np.where(np.isin(RALL, z['test']))[0]; assert (RALL[te] == z['ridt']).all()
    for kmin in (2, 5): r[f'M0late{kmin}'], r[f'M5_{kmin}'] = m5(z['Zt'], z['yt'], z['ridt'], TALL[te], T, b, pi_s, kmin)
    r['split'] = int(f.split('_')[-1][:-4]); r['test'] = z['test'].tolist(); sp.append(r)
N['splits'] = sp
json.dump(N, open('res/v3_numbers.json', 'w'), indent=1, default=float)

# ---- console summary ----
def line(tag, runs, get):
    for m in get(runs[0]):
        if m not in METHODS: continue
        g = lambda key: summarise(runs, lambda r: get(r)[m][key] if key in ('kappa', 'acc', 'score', 'macroF1') else get(r)[m][key[0]][key[1]])
        print(f"{tag:12s} {m}: kappa {g('kappa')['mean']:.3f}±{g('kappa')['sd']:.3f}  S Se/PPV {g(('S','se'))['mean']:.3f}/{g(('S','ppv'))['mean']:.3f} (Se sd {g(('S','se'))['sd']:.3f})"
              f"  V Se/PPV {g(('V','se'))['mean']:.3f}/{g(('V','ppv'))['mean']:.3f}  F Se {g(('F','se'))['mean']:.3f}  macroF1 {g('macroF1')['mean']:.3f}")
line('v2 DS2', runs, lambda r: r)
line('causal DS2', runs_c, lambda r: r['ds2'])
line('causal INC', runs_c, lambda r: r['incart'])
if 'ds2_int8' in runs_c[0]:
    rq = [r for r in runs_c if 'ds2_int8' in r]
    line('INT8 DS2', rq, lambda r: r['ds2_int8']); line('TFLM DS2', rq, lambda r: r['ds2_tflm']); line('INT8 INC', rq, lambda r: r['incart_int8'])
if sp:
    line('splits', sp, lambda r: r)
    for kmin in (2, 5):
        d = np.array([r[f'M5_{kmin}']['score'] - r[f'M0late{kmin}']['score'] for r in sp]); dk = np.array([r[f'M5_{kmin}']['kappa'] - r[f'M0late{kmin}']['kappa'] for r in sp])
        print(f'splits M5_{kmin} - M0 (same beats): score {d.mean():+.3f} (SD {d.std(ddof=1):.3f}) wins {(d > 0).sum()}/{len(d)} | kappa {dk.mean():+.3f} wins {(dk > 0).sum()}/{len(dk)}')
    for m in ('M2', 'M4'):
        d = np.array([r[m]['score'] - r['M0']['score'] for r in sp]); print(f'splits {m} - M0: score {d.mean():+.3f} (SD {d.std(ddof=1):.3f}) wins {(d > 0).sum()}/{len(d)}')
