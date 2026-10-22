"""Figures for v3: variance strip plot, EM vs true S prevalence, causal confusion matrix."""
import sys, json, glob; sys.path.insert(0, '.'); from lib import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
OUT = '/home/claude/work/paper'; N = json.load(open('res/v3_numbers.json'))
C = ['#2a78d6', '#eb6834', '#1baf7a', '#8a5cc2', '#6b6b6b']

# ---- Fig 2: sources of variance in S sensitivity / PPV ----
D2 = cached('v2_ds2', None); byseed = {r['seed']: r['M0'] for r in N['v2_ds2']}
off, on = [], []
for s_ in (42, 1, 2, 3, 4):
    f = f'res/final_v2dnn_s{s_}.npz'
    if os.path.exists(f): off.append(byseed[s_]); on.append(metrics(D2['y'], np.load(f)['Zt'].argmax(1)))
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), dpi=200); rng = np.random.RandomState(0)
for ax, key, lab in ((axs[0], 'se', 'S sensitivity'), (axs[1], 'ppv', 'S precision (PPV)')):
    v = np.array([r['M0']['S'][key] for r in N['v2_ds2']]); ax.scatter(rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=C[0], edgecolor='none')
    ax.plot([-0.25, 0.25], [np.median(v)] * 2, color='#222', lw=1.2)
    a = np.array([r['S'][key] for r in off]); b = np.array([r['S'][key] for r in on])
    for i in range(len(a)): ax.plot([0.85, 1.15], [a[i], b[i]], color='#9a9a9a', lw=0.8)
    ax.scatter(np.full(len(a), 0.85), a, s=14, color=C[0], edgecolor='none'); ax.scatter(np.full(len(b), 1.15), b, s=14, color=C[1], edgecolor='none')
    v = np.array([r['M0']['S'][key] for r in N['splits']]); ax.scatter(2 + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=C[2], edgecolor='none')
    ax.plot([1.75, 2.25], [np.median(v)] * 2, color='#222', lw=1.2)
    ax.set_xticks([0, 1, 2], ['DS2, 10 training\nseeds', 'DS2, same seed:\noneDNN off / on', '20 random\npatient splits'], fontsize=7)
    ax.set_ylim(0, 1); ax.set_xlim(-0.5, 2.5); ax.set_ylabel(lab); ax.grid(axis='y', color='#e6e5e0', lw=0.6); ax.set_axisbelow(True)
    for sp_ in ('top', 'right'): ax.spines[sp_].set_visible(False)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_variance.png', dpi=300, facecolor='white')

# ---- Fig 3: label-free (offline EM) estimate of the S prevalence vs truth, per record ----
fig, ax = plt.subplots(figsize=(3.6, 3.2), dpi=200)
for mode, col, lab in (('cv', C[0], 'DS1 CV folds'), ('ds2', C[1], 'DS2')):
    p = f'res/prior_{mode}.json'
    if not os.path.exists(p): continue
    d = json.load(open(p))['diag']; x = np.array([r['true_S'] for r in d]); y = np.array([r['em_S'] for r in d])
    ax.scatter(np.maximum(x, 1e-4), np.maximum(y, 1e-4), s=12, color=col, alpha=0.7, edgecolor='none', label=lab)
ax.plot([1e-4, 1.2], [1e-4, 1.2], color='#b5b4ae', lw=1, ls=(0, (2, 2)))
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(8e-5, 1.2); ax.set_ylim(8e-5, 1.2)
ax.set_xlabel('True S fraction in record (0 plotted at $10^{-4}$)'); ax.set_ylabel('Label-free EM estimate')
ax.legend(frameon=False, fontsize=7, loc='upper left'); ax.grid(color='#e6e5e0', lw=0.6); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_prior.png', dpi=300, facecolor='white')

# ---- Fig 4: confusion matrix, causal front end, median-kappa seed ----
C2 = cached('c_ds2', None); fs = sorted(glob.glob('res/final_causal_s*.npz')); m = C2['y'] >= 0
ks = sorted((cohen_kappa_score(C2['y'][m], np.load(f)['Zt'][m].argmax(1)), f) for f in fs); kap, f = ks[len(ks) // 2]
y = C2['y'][m]; pr = np.load(f)['Zt'][m].argmax(1)
cm = np.array([[((y == i) & (pr == j)).sum() for j in range(4)] for i in range(4)], float); cmn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(4.2, 3.7), dpi=200); im = ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f'{cmn[i, j]:.2f}\n({int(cm[i, j]):,})', ha='center', va='center', fontsize=7.5, color='white' if cmn[i, j] > 0.55 else '#0b0b0b')
ax.set_xticks(range(4), NAMES); ax.set_yticks(range(4), [f'{n} ({int(cm[i].sum()):,})' for i, n in enumerate(NAMES)])
ax.set_xlabel('Predicted class'); ax.set_ylabel('True class (support)')
for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, fraction=0.046, pad=0.04); cb.outline.set_visible(False); cb.set_label('Fraction of true class', fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_confusion_causal.png', dpi=300, facecolor='white')
print('median-kappa causal seed', os.path.basename(f), round(kap, 4))
