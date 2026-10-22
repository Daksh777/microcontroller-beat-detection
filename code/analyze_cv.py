"""Apply the pre-registered lambda selection rule on DS1 CV (no DS2 data touched)."""
import sys, glob; sys.path.insert(0, '.'); from lib import *
LAMS = [1/50, 1/100, 1/200, 1/400, 1/800]

def prep_cal(Zv, yv):
    T, b = fit_bcts(Zv, yv[yv >= 0] if False else yv)  # val y always >=0 in v2 data
    pi_s = np.maximum(np.bincount(yv[yv >= 0], minlength=NC) / (yv >= 0).sum(), 1e-4); pi_s /= pi_s.sum()
    return T, b, pi_s

cells = {}
for f in sorted(glob.glob('res/cv_f*_s*.npz')):
    z = np.load(f); T, b, pi_s = prep_cal(z['Zv'], z['yv'])
    row = {'T': T, 'b': b.round(3).tolist(), 'pi_s': pi_s.round(4).tolist()}
    for m in ('M0', 'M1', 'M3', 'M4', 'M3c', 'M4c'):
        row[m] = metrics(z['yt'], apply_method(m, z['Zt'], z['yt'], z['ridt'], T, b, pi_s).argmax(1))['score']
    for lam in LAMS:
        row[f'M2_{lam:.4f}'] = metrics(z['yt'], apply_method('M2', z['Zt'], z['yt'], z['ridt'], T, b, pi_s, lam).argmax(1))['score']
        row[f'M2c_{lam:.4f}'] = metrics(z['yt'], apply_method('M2c', z['Zt'], z['yt'], z['ridt'], T, b, pi_s, lam).argmax(1))['score']
    cells[os.path.basename(f)[:-4]] = row
assert len(cells) == 8, len(cells)
keys = ['M0', 'M1', 'M3', 'M4', 'M3c', 'M4c'] + [f'M2_{l:.4f}' for l in LAMS] + [f'M2c_{l:.4f}' for l in LAMS]
summ = {k: dict(mean=float(np.mean([c[k] for c in cells.values()])), se=float(np.std([c[k] for c in cells.values()], ddof=1) / np.sqrt(8))) for k in keys}
for k in keys: print(f"{k:10s} {summ[k]['mean']:.4f} +- {summ[k]['se']:.4f}")
m2 = [f'M2_{l:.4f}' for l in LAMS]; best = max(m2, key=lambda k: summ[k]['mean'])
ok = [k for k in m2 if summ[k]['mean'] >= summ[best]['mean'] - summ[best]['se']]
L = {f'M2_{l:.4f}': l for l in LAMS}; chosen = min(ok, key=lambda k: L[k])   # smallest lambda = slowest adaptation
d = np.array([cells[c][chosen] - cells[c]['M0'] for c in cells]); pse = d.std(ddof=1) / np.sqrt(len(d))
adopt = bool(d.mean() > pse)
print('best', best, '| chosen', chosen, f'| paired diff vs M0 {d.mean():+.4f} (paired SE {pse:.4f}), wins {int((d > 0).sum())}/8 -> adopt={adopt}')
def rule(prefix):
    m2 = [f'{prefix}_{l:.4f}' for l in LAMS]; best = max(m2, key=lambda k: summ[k]['mean'])
    ok = [k for k in m2 if summ[k]['mean'] >= summ[best]['mean'] - summ[best]['se']]
    Lc = {f'{prefix}_{l:.4f}': l for l in LAMS}; ch = min(ok, key=lambda k: Lc[k])
    dd = np.array([cells[c][ch] - cells[c]['M0'] for c in cells]); se_ = dd.std(ddof=1) / np.sqrt(len(dd))
    print(prefix, 'best', best, 'chosen', ch, f'diff {dd.mean():+.4f} SE {se_:.4f} wins {int((dd > 0).sum())}/8 adopt={dd.mean() > se_}')
    return dict(best=best, chosen=ch, lam=Lc[ch], paired_diff=float(dd.mean()), paired_se=float(se_), wins=int((dd > 0).sum()), adopt=bool(dd.mean() > se_))
A1 = rule('M2c')
json.dump(dict(A1=A1, cells=cells, summary=summ, best=best, chosen=chosen, lam=L[chosen], paired_diff=float(d.mean()), paired_se=float(pse),
               wins=int((d > 0).sum()), adopt=adopt), open('res/cv_selection.json', 'w'), indent=1)
