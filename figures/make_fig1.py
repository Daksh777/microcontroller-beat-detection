import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
INK = '#222222'; C_DATA = '#E8EEF6'; C_PROC = '#F3F3F3'; C_ECG = '#DCE9F5'; C_RR = '#F8E6D2'; C_FUSE = '#E4F0E0'; C_DEP = '#EDE4F3'
fig = plt.figure(figsize=(7.2, 5.6), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 78); ax.axis('off')
def box(x, y, w, h, text, fc, bold=False, fs=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=0.8', fc=fc, ec=INK, lw=0.7))
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs, color=INK, weight='bold' if bold else 'normal', linespacing=1.25)
def arr(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=7, lw=0.7, color=INK, shrinkA=0, shrinkB=0))
# ---------- (a) pipeline ----------
ax.text(1, 76, '(a) Processing and evaluation pipeline', fontsize=8.5, weight='bold', va='center')
Y = 60; H = 11
steps = [(1, 'MIT-BIH\nArrhythmia DB\nlead MLII, 360 Hz\n44 non-paced recs', C_DATA),
         (17.2, 'Causal filtering\n0.5–45 Hz band-pass\n(4th order, 1 pass)\n+ 60 Hz notch', C_PROC),
         (33.4, 'Causal Pan–Tompkins\nQRS detector;\n0.5-s beat window\n+ 5 causal RR feats', C_PROC),
         (49.6, 'Inter-patient split\nDS1: 17 train recs\n+ 5 val recs\nDS2: 22 test recs', C_DATA),
         (65.8, 'Dual-branch CNN\n924 params\n(+ optional per-\npatient prior)', C_FUSE),
         (82.0, 'Full-integer INT8\n(TFLite)\nrun in TFLite Micro\nruntime', C_DEP)]
W = 15.2
for i, (x, t, c) in enumerate(steps):
    box(x, Y, W, H, t, c)
    if i: arr(steps[i-1][0] + W + 0.35, Y + H/2, x - 0.35, Y + H/2)
ax.text(57.2, Y - 2.6, 'DS2 used only for reporting; architecture and all free parameters chosen on DS1; analyses pre-registered', ha='center', fontsize=6.6, style='italic', color='#444444')
# ---------- (b) architecture ----------
ax.text(1, 52, '(b) Model (0.15× width, 924 parameters)', fontsize=8.5, weight='bold', va='center')
# ECG branch row
ye, h = 38, 8
ecg = [('ECG beat\n180 × 1', C_DATA, 9), ('Conv 5, k7\nReLU\n180 × 5', C_ECG, 9), ('MaxPool 2\n90 × 5', C_PROC, 7.5), ('Conv 5, k5\nReLU\n90 × 5', C_ECG, 9),
       ('MaxPool 2\n45 × 5', C_PROC, 7.5), ('Conv 10, k3\nReLU\n45 × 10', C_ECG, 9), ('MaxPool 2\n22 × 10', C_PROC, 7.5), ('Conv 10, k3\nReLU\n22 × 10', C_ECG, 9), ('Global\navg pool\n10', C_ECG, 7.5)]
x = 1; xs = []
for t, c, w in ecg:
    box(x, ye, w, h, t, c, fs=6.6); xs.append((x, w)); x += w + 2.2
for i in range(1, len(xs)):
    arr(xs[i-1][0] + xs[i-1][1] + 0.3, ye + h/2, xs[i][0] - 0.3, ye + h/2)
gx, gw = xs[-1]
# RR branch row
yr = 22
rr = [('RR features\n5', C_DATA, 9), ('Batch\nNorm', C_PROC, 7.5), ('Dense 5\nReLU', C_RR, 8), ('Dense 5\nReLU', C_RR, 8)]
x = 1; rs = []
for t, c, w in rr:
    box(x, yr, w, h, t, c, fs=6.6); rs.append((x, w)); x += w + 2.2
for i in range(1, len(rs)):
    arr(rs[i-1][0] + rs[i-1][1] + 0.3, yr + h/2, rs[i][0] - 0.3, yr + h/2)
ax.text(1, ye + h + 1.6, 'Morphology branch', fontsize=7, style='italic', color='#333333')
ax.text(1, yr + h + 1.6, 'Timing branch: pre-RR, post-RR (1-beat delay), pre/post ratio, local RR (10), prev-2 RR; all ÷ causal mean of ≤80 past RRs', fontsize=6.4, style='italic', color='#333333')
# fusion column
fx = 36; fy = 6; fw = 10
box(fx, fy, fw, 8, 'Concat\n10 + 5 = 15', C_FUSE, fs=6.6)
arr(gx + gw/2, ye - 0.3, fx + fw*0.7, fy + 8.3)
arr(rs[-1][0] + rs[-1][1]/2, yr - 0.3, fx + fw*0.25, fy + 8.3)
seq = [('Dropout 0.4', 8), ('Dense 10\nReLU', 7.5), ('Dropout 0.3', 8), ('Softmax 4\nN / S / V / F', 11)]
x = fx + fw + 2.2; prev = (fx, fw)
for t, w in seq:
    box(x, fy, w, 8, t, C_FUSE, fs=6.4); arr(prev[0] + prev[1] + 0.3, fy + 4, x - 0.3, fy + 4); prev = (x, w); x += w + 2.2
ax.text(1, 3.2, 'No residual connections, batch normalisation in the conv branch, or squeeze-and-excitation blocks.  31,140 multiply-accumulates per beat;\nfull-integer INT8 model 9.5 KB flash, 5.9 KB TFLite Micro tensor arena (causal pipeline).', fontsize=6.4, color='#333333', va='center')
fig.savefig('fig1.png', dpi=300, facecolor='white'); print('ok')
