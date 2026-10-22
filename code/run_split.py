"""Random patient-wise split i of the 44 records: 22 test, 17 train + 5 val (v2 front end)."""
import sys; sys.path.insert(0, '.'); from lib import *
i = int(sys.argv[1]); out = f'res/split_{i}.npz'
if os.path.exists(out): sys.exit()
A = cached('v2_ds1', None); B = cached('v2_ds2', None); D = {k: np.concatenate([A[k], B[k]]) for k in A}
recs = np.array(DS1 + DS2); rng = np.random.RandomState(i); p = rng.permutation(len(recs))
test, val, trn = recs[p[:22]], recs[p[22:27]], recs[p[27:]]
te, va, tr = (np.where(np.isin(D['rid'], s))[0] for s in (test, val, trn))
m, lm, fmt, secs = train(i, D['X'][tr], D['y'][tr], D['R'][tr], D['X'][va], D['y'][va], D['R'][va])
P = lambda j: lm.predict([fmt(D['X'][j]), D['R'][j]], batch_size=4096, verbose=0)
np.savez_compressed(out, Zv=P(va), yv=D['y'][va], ridv=D['rid'][va], Zt=P(te), yt=D['y'][te], ridt=D['rid'][te], secs=secs, test=test, val=val)
print('split', i, f'{secs:.0f}s', flush=True)
