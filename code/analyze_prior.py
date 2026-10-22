"""A2: few-label prior (first k minutes) and D1 diagnostic of label-free prior estimation. Usage: analyze_prior.py cv|ds2"""
import sys, glob; sys.path.insert(0, '.'); from lib import *
mode = sys.argv[1]
def cal(Zv, yv):
    T, b = fit_bcts(Zv, yv); pi_s = np.maximum(np.bincount(yv, minlength=NC) / len(yv), 1e-4); return T, b, pi_s / pi_s.sum()
def true_prior(y): p = np.maximum(np.bincount(y, minlength=NC) / len(y), 1e-4); return p / p.sum()
cells = []
if mode == 'cv':
    D = cached('v2_ds1', None); t_all = np.load('cache/v2_ds1_t.npy')
    items = [(f, np.where(np.isin(D['rid'], FOLDS[int(f.split('_f')[1][0])]))[0]) for f in sorted(glob.glob('res/cv_f*_s*.npz'))]
else:
    D = cached('v2_ds2', None); t_all = np.load('cache/v2_ds2_t.npy')
    items = [(f, np.arange(len(D['y']))) for f in sorted(glob.glob('res/final_v2_s*.npz'))]
diag = []
for f, idx in items:
    z = np.load(f); Zt = z['Zt']; y = D['y'][idx]; rid = D['rid'][idx]; t = t_all[idx]
    T, b, pi_s = cal(z['Zv'], z['yv']); Q = calibrated(Zt, T, b); S = np.exp(log_softmax(Zt))
    row = {}
    for k in (2, 5):
        P4 = np.empty_like(Q); P5 = np.empty_like(Q); late = t >= k * 60 * 360
        for rec in dict.fromkeys(rid):
            j = np.where(rid == rec)[0]; e = j[t[j] < k * 60 * 360]
            pk = true_prior(y[e]) if len(e) else pi_s
            p = Q[j] * pk / pi_s; P5[j] = p / p.sum(1, keepdims=True)
            p = Q[j] * true_prior(y[j]) / pi_s; P4[j] = p / p.sum(1, keepdims=True)
        for nm, P in (('M0', S), ('M4', P4), (f'M5', P5)):
            row[f'{nm}_k{k}'] = metrics(y[late], P[late].argmax(1))
    for rec in dict.fromkeys(rid):
        j = rid == rec; diag.append(dict(file=os.path.basename(f), rec=rec, true_S=float((y[j] == 1).mean()), em_S=float(offline_prior(Q[j], pi_s)[1]),
                                         em_V=float(offline_prior(Q[j], pi_s)[2]), true_V=float((y[j] == 2).mean()), pi_s_S=float(pi_s[1])))
    cells.append(row)
for k in (2, 5):
    for nm in ('M0', 'M4', 'M5'):
        sc = np.array([c[f'{nm}_k{k}']['score'] for c in cells]); ka = np.array([c[f'{nm}_k{k}']['kappa'] for c in cells])
        Sse = np.array([c[f'{nm}_k{k}']['S']['se'] for c in cells]); Spp = np.array([c[f'{nm}_k{k}']['S']['ppv'] for c in cells])
        print(f'k={k} {nm}: score {sc.mean():.3f}±{sc.std(ddof=1)/np.sqrt(len(sc)):.3f} kappa {ka.mean():.3f} S Se/PPV {Sse.mean():.3f}/{Spp.mean():.3f}')
    d = np.array([c[f'M5_k{k}']['score'] - c[f'M0_k{k}']['score'] for c in cells]); print(f'   M5-M0 paired {d.mean():+.3f} (SE {d.std(ddof=1)/np.sqrt(len(d)):.3f}) wins {(d>0).sum()}/{len(d)}')
dd = [x for x in diag]; noS = [x for x in dd if x['true_S'] < 0.002]
print('D1: records with true S<0.2%:', len(noS), 'median EM S prior', np.median([x['em_S'] for x in noS]).round(4), 'vs pi_s median', np.median([x['pi_s_S'] for x in noS]).round(4))
hiS = [x for x in dd if x['true_S'] >= 0.05]
for x in hiS[:12]: print('   ', x['file'], x['rec'], 'true S %.3f EM S %.3f' % (x['true_S'], x['em_S']))
print('corr(true_S, em_S)', np.corrcoef([x['true_S'] for x in dd], [x['em_S'] for x in dd])[0, 1].round(3), '| V', np.corrcoef([x['true_V'] for x in dd], [x['em_V'] for x in dd])[0, 1].round(3))
json.dump(dict(cells=cells, diag=diag), open(f'res/prior_{mode}.json', 'w'), indent=1, default=float)
