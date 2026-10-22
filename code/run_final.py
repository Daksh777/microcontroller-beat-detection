"""Final DS2 models (train 17 DS1 recs, val = 5 VAL_RECS), saving logits for val, DS2 (and INCART for causal)."""
import sys; sys.path.insert(0, '.'); from lib import *
pipe, seed = sys.argv[1], int(sys.argv[2])   # pipe: v2 | causal
out = f'res/final_{pipe}{os.environ.get("TAG","")}_s{seed}.npz'
if os.path.exists(out): sys.exit()
D1 = cached('v2_ds1' if pipe == 'v2' else 'c_ds1', None); D2 = cached('v2_ds2' if pipe == 'v2' else 'c_ds2', None)
lab = D1['y'] >= 0; va = np.isin(D1['rid'], VAL_RECS)
tr = np.where(lab & ~va)[0]; vi = np.where(va)[0]          # val keeps all detections (stream order) ; scoring uses y>=0
vs = vi[D1['y'][vi] >= 0]
m, lm, fmt, secs = train(seed, D1['X'][tr], D1['y'][tr], D1['R'][tr], D1['X'][vs], D1['y'][vs], D1['R'][vs], conv2d=(pipe == 'causal'))
Zv = lm.predict([fmt(D1['X'][vi]), D1['R'][vi]], batch_size=4096, verbose=0)
Zt = lm.predict([fmt(D2['X']), D2['R']], batch_size=4096, verbose=0)
extra = {}
if pipe == 'causal':
    DI = cached('c_incart', None); extra['Zi'] = lm.predict([fmt(DI['X']), DI['R']], batch_size=4096, verbose=0)
    m.save(f'res/final_{pipe}_s{seed}.keras')
np.savez_compressed(out, Zv=Zv, yv=D1['y'][vi], ridv=D1['rid'][vi], Zt=Zt, secs=secs, params=m.count_params(), **extra)
r = metrics(D2['y'], Zt.argmax(1)); print(pipe, seed, f'{secs:.0f}s', 'kappa', r['kappa'], 'S', r['S'], 'V', r['V'], flush=True)
