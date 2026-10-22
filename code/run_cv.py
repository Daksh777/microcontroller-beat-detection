"""DS1 CV for choosing lambda: held-out fold k scored; fold (k+1)%4 = val (checkpoint + BCTS); other two folds train."""
import sys; sys.path.insert(0, '.'); from lib import *
k, seed = int(sys.argv[1]), int(sys.argv[2]); out = f'res/cv_f{k}_s{seed}.npz'
if os.path.exists(out): sys.exit()
D1 = cached('v2_ds1', None); rid = D1['rid']
te = np.where(np.isin(rid, FOLDS[k]))[0]; va = np.where(np.isin(rid, FOLDS[(k + 1) % 4]))[0]
tr = np.where(~np.isin(rid, FOLDS[k] + FOLDS[(k + 1) % 4]))[0]
m, lm, fmt, secs = train(seed, D1['X'][tr], D1['y'][tr], D1['R'][tr], D1['X'][va], D1['y'][va], D1['R'][va])
P = lambda i: lm.predict([fmt(D1['X'][i]), D1['R'][i]], batch_size=4096, verbose=0)
np.savez_compressed(out, Zv=P(va), yv=D1['y'][va], ridv=rid[va], Zt=P(te), yt=D1['y'][te], ridt=rid[te], secs=secs)
print('cv', k, seed, f'{secs:.0f}s', flush=True)
