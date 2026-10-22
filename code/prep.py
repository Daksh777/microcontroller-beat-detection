import sys; sys.path.insert(0, '.'); from lib import *
v1 = cached('v2_ds1', lambda: load_v2(DS1)); v2 = cached('v2_ds2', lambda: load_v2(DS2))
print('v2 DS1', len(v1['y']), np.bincount(v1['y']), 'DS2', len(v2['y']), np.bincount(v2['y']))
for nm, recs in (('c_ds1', DS1), ('c_ds2', DS2)):
    d = cached(nm, lambda: load_causal(recs)); st = json.load(open(f'cache/{nm}_stats.json'))
    tp = sum(s['tp'] for s in st.values()); nb = sum(s['n_beats'] for s in st.values()); fp = sum(s['fp'] for s in st.values())
    print(nm, 'detector Se %.4f PPV %.4f' % (tp / nb, tp / (tp + fp)), 'scored beats', np.bincount(d['y'][d['y'] >= 0]), 'fp', (d['y'] == -1).sum(), 'unscored', (d['y'] == -2).sum())
    for c in NAMES: print('  missed', c, sum(s['missed_by_class'][c] for s in st.values()), '/', sum(s['total_by_class'][c] for s in st.values()))
