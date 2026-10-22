import sys; sys.path.insert(0, '.'); from lib import *
import wfdb
def times(recs):
    out = []
    for rec in recs:
        p = os.path.join(MITDB, rec); r = wfdb.rdrecord(p, sampto=1); a = wfdb.rdann(p, 'atr'); n = wfdb.rdheader(p).sig_len
        for sm, c in zip(a.sample, a.symbol):
            if c in AAMI and sm >= WIN and sm + WIN <= n: out.append(sm)
    return np.array(out)
for nm, recs in (('v2_ds1', DS1), ('v2_ds2', DS2)):
    t = times(recs); d = np.load(f'cache/{nm}.npz'); assert len(t) == len(d['y']); np.save(f'cache/{nm}_t.npy', t); print(nm, len(t))
