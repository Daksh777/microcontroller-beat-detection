"""v3 library: data (v2 pipeline + causal real-time front end), model, training, calibration and adaptation."""
import os, json, time, numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfilt, lfilter, resample_poly, find_peaks
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
from scipy.optimize import minimize

MITDB = '/home/claude/data/mit-bih-arrhythmia-database-1.0.0'
INCART = '/home/claude/data/st-petersburg-incart-12-lead-arrhythmia-database-1.0.0/files'
CACHE = '/home/claude/work/v3/cache'; os.makedirs(CACHE, exist_ok=True)
NAMES = ['N', 'S', 'V', 'F']; NC = 4
AAMI = {'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0, 'A': 1, 'a': 1, 'J': 1, 'S': 1, 'V': 2, 'E': 2, 'F': 3}
BEATSYM = set('NLRBAaJSVrFejnE/fQ?')
DS1 = ['101','106','108','109','112','114','115','116','118','119','122','124','201','203','205','207','208','209','215','220','223','230']
DS2 = ['100','103','105','111','113','117','121','123','200','202','210','212','213','214','219','221','222','228','231','232','233','234']
VAL_RECS = ['106', '118', '124', '201', '223']
FOLDS = [['209','106','116','101','112','230'], ['201','220','119','108','115','205'],
         ['207','118','203','109','122','215'], ['223','124','208','114']]
WIN = 180

def znorm(s):
    sd = s.std(); return (s - s.mean()) / sd if sd > 1e-8 else s - s.mean()

def rr_new(bs, j):
    rr = np.diff(bs).astype(np.float64)
    pre = rr[j-1] if j >= 1 else np.nan; post = rr[j] if j < len(rr) else np.nan; prev2 = rr[j-2] if j >= 2 else np.nan
    hist = rr[max(0, j-80):j]
    if len(hist) == 0: return [1.0] * 5
    m_long = hist.mean(); m10 = hist[-10:].mean()
    f = lambda v: 1.0 if np.isnan(v) else float(v / m_long)
    ratio = 1.0 if (np.isnan(pre) or np.isnan(post) or post <= 0) else pre / post
    return [min(f(pre), 5), min(f(post), 5), min(ratio, 5), float(m10 / m_long), min(f(prev2), 5)]

# ---------------- v2 (offline) front end: identical to the paper ----------------
def preprocess_v2(raw, fs):
    b, a = iirnotch(50.0 / (0.5 * fs), 30.0); raw = filtfilt(b, a, raw)
    b, a = butter(4, [0.5 / (0.5 * fs), 45.0 / (0.5 * fs)], btype='band'); return filtfilt(b, a, raw)

def load_v2(recs):
    import wfdb
    X, y, R, rid = [], [], [], []
    for rec in recs:
        p = os.path.join(MITDB, rec); r = wfdb.rdrecord(p); a = wfdb.rdann(p, 'atr')
        sig = preprocess_v2(r.p_signal[:, r.sig_name.index('MLII')], r.fs)
        s, sym = a.sample, np.array(a.symbol)
        keep = np.array([c in BEATSYM for c in sym]); bs = s[keep]; pos = {v: i for i, v in enumerate(bs)}
        for sm, c in zip(s, sym):
            if c not in AAMI or sm < WIN or sm + WIN > len(sig): continue
            X.append(znorm(sig[sm-WIN:sm+WIN])[90:270]); y.append(AAMI[c]); R.append(rr_new(bs, pos[sm])); rid.append(rec)
    return dict(X=np.array(X, np.float32), y=np.array(y, np.int32), R=np.array(R, np.float32), rid=np.array(rid))

# ---------------- causal real-time front end ----------------
def causal_filter(raw, fs, notch=60.0):
    sos = butter(4, [0.5, 45.0], btype='band', fs=fs, output='sos'); x = sosfilt(sos, raw - raw[0])
    if notch:
        b, a = iirnotch(notch, 30.0, fs=fs); x = lfilter(b, a, x)
    return x

def pan_tompkins(x, fs):
    """Causal Pan-Tompkins-style detector on an already band-limited ECG x. Returns R-peak sample indices.
    Every decision uses past samples plus a bounded look-ahead (peak confirmation <=200 ms; search-back <=1.66 RR)."""
    sos = butter(2, [5, 15], btype='band', fs=fs, output='sos'); b = sosfilt(sos, x)
    d = np.zeros_like(b); d[4:] = (2*b[4:] + b[3:-1] - b[1:-3] - 2*b[:-4]) / 8.0
    sq = d ** 2; w = int(0.150 * fs); mwi = np.convolve(sq, np.ones(w) / w)[:len(sq)]
    cand, _ = find_peaks(mwi, distance=int(0.2 * fs))
    init = int(2 * fs); spki = 0.25 * mwi[:init].max(); npki = 0.5 * mwi[:init].mean()
    qrs = []; slopes = []; rr_hist = []; last_idx = None
    def thr(): return npki + 0.25 * (spki - npki)
    i = 0; pending = []   # candidates seen since last QRS (for search-back)
    while i < len(cand):
        c = cand[i]; pk = mwi[c]
        # search-back: if too long since last QRS, look back for the biggest candidate above thr/2
        if qrs and rr_hist:
            rr_avg = np.mean(rr_hist[-8:])
            if c - qrs[-1] > 1.66 * rr_avg and pending:
                best = max(pending, key=lambda k: mwi[k])
                if mwi[best] > 0.5 * thr():
                    qrs.append(best); slopes.append(np.abs(d[max(0, best - w):best + 1]).max())
                    rr_hist.append(qrs[-1] - qrs[-2]); spki = 0.25 * mwi[best] + 0.75 * spki
                    pending = [k for k in pending if k > best + int(0.2 * fs)]
                    continue
        if pk > thr():
            sl = np.abs(d[max(0, c - w):c + 1]).max()
            if qrs and c - qrs[-1] < int(0.36 * fs) and sl < 0.5 * slopes[-1]:
                npki = 0.125 * pk + 0.875 * npki; pending.append(c)   # T-wave
            else:
                if qrs: rr_hist.append(c - qrs[-1])
                qrs.append(c); slopes.append(sl); spki = 0.125 * pk + 0.875 * spki; pending = []
        else:
            npki = 0.125 * pk + 0.875 * npki; pending.append(c)
        i += 1
    # R-peak location: largest |x| deviation in the 250 ms before the MWI peak (causal w.r.t. the detection)
    lb = int(0.25 * fs); out = []
    for c in qrs:
        seg = x[max(0, c - lb):c + 1]; base = np.median(seg)
        out.append(max(0, c - lb) + int(np.argmax(np.abs(seg - base))))
    out = np.unique(np.array(out, np.int64))
    if len(out) > 1:   # merge duplicates closer than 200 ms (keep first)
        keep = [out[0]]
        for v in out[1:]:
            if v - keep[-1] >= int(0.2 * fs): keep.append(v)
        out = np.array(keep)
    return out

def match(det, ann_s, ann_sym, fs, tol=0.150):
    """Match each detection to the nearest beat annotation within tol. Returns label per detection
    (0-3 AAMI, -2 = matched non-scored beat (Q/paced), -1 = false detection), and the set of matched annotation indices."""
    beat_idx = np.array([k for k, c in enumerate(ann_sym) if c in BEATSYM]); bs = ann_s[beat_idx]
    lab = np.full(len(det), -1, np.int32); used = set(); t = int(tol * fs)
    j = np.searchsorted(bs, det)
    for i, dv in enumerate(det):
        best, bd = None, t + 1
        for k in (j[i] - 1, j[i]):
            if 0 <= k < len(bs) and abs(int(bs[k]) - int(dv)) < bd and k not in used: best, bd = k, abs(int(bs[k]) - int(dv))
        if best is not None:
            used.add(best); c = ann_sym[beat_idx[best]]; lab[i] = AAMI.get(c, -2)
    return lab, used, beat_idx

def load_causal(recs, db='mitdb'):
    import wfdb
    X, y, R, rid = [], [], [], []; stats = {}
    for rec in recs:
        p = os.path.join(MITDB if db == 'mitdb' else INCART, rec); r = wfdb.rdrecord(p); a = wfdb.rdann(p, 'atr')
        if db == 'mitdb':
            raw = r.p_signal[:, r.sig_name.index('MLII')]; s = a.sample; fs = r.fs
        else:
            raw = resample_poly(r.p_signal[:, r.sig_name.index('II')], 360, r.fs); s = np.round(a.sample * 360 / r.fs).astype(np.int64); fs = 360
        sym = np.array(a.symbol); x = causal_filter(raw, fs)
        det = pan_tompkins(x, fs); lab, used, beat_idx = match(det, s, sym, fs)
        n_ann_scored = np.array([AAMI.get(sym[k], -2) for k in beat_idx]); missed = np.array([k not in used for k in range(len(beat_idx))])
        stats[rec] = dict(n_det=len(det), n_beats=len(beat_idx), tp=int(len(used)), fp=int((lab == -1).sum()),
                          missed_by_class={NAMES[c]: int(((n_ann_scored == c) & missed).sum()) for c in range(NC)},
                          total_by_class={NAMES[c]: int((n_ann_scored == c).sum()) for c in range(NC)})
        for j, dv in enumerate(det):
            if dv < WIN or dv + WIN > len(x): continue
            X.append(znorm(x[dv-WIN:dv+WIN])[90:270]); y.append(lab[j]); R.append(rr_new(det, j)); rid.append(rec)
    return dict(X=np.array(X, np.float32), y=np.array(y, np.int32), R=np.array(R, np.float32), rid=np.array(rid)), stats

def cached(name, fn):
    p = os.path.join(CACHE, name + '.npz')
    if not os.path.exists(p):
        out = fn(); stats = None
        if isinstance(out, tuple): out, stats = out
        np.savez_compressed(p, **out)
        if stats is not None: json.dump(stats, open(p[:-4] + '_stats.json', 'w'), indent=1)
    return dict(np.load(p))

# ---------------- model + training (v2 recipe, logits exposed) ----------------
def build(width=0.15, conv2d=False, alpha=None):
    import tensorflow as tf; from tensorflow import keras; from tensorflow.keras import layers
    wf = lambda n: max(4, int(round(n * width)))
    if conv2d:
        e = keras.Input(shape=(180, 1, 1), name='ecg_input')
        conv = lambda f, k: layers.Conv2D(f, (k, 1), padding='same', activation='relu'); pool = lambda: layers.MaxPooling2D((2, 1)); gap = layers.GlobalAveragePooling2D
    else:
        e = keras.Input(shape=(180, 1), name='ecg_input')
        conv = lambda f, k: layers.Conv1D(f, k, padding='same', activation='relu'); pool = lambda: layers.MaxPooling1D(2); gap = layers.GlobalAveragePooling1D
    x = conv(wf(32), 7)(e); x = pool()(x); x = conv(wf(32), 5)(x); x = pool()(x); x = conv(wf(64), 3)(x); x = pool()(x); x = conv(wf(64), 3)(x)
    ef = gap()(x)
    r = keras.Input(shape=(5,), name='rr_input'); rn = layers.BatchNormalization()(r)
    rf = layers.Dense(wf(32), activation='relu')(layers.Dense(wf(32), activation='relu')(rn))
    f = layers.Dropout(0.4)(layers.Concatenate()([ef, rf])); f = layers.Dense(wf(64), activation='relu')(f); f = layers.Dropout(0.3)(f)
    logits = layers.Dense(NC, name='logits')(f); out = layers.Activation('softmax')(logits)
    at = tf.constant(alpha, tf.float32)
    def fl(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32); y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        oh = tf.one_hot(y_true, NC); pt = tf.reduce_sum(oh * y_pred, -1)
        return tf.reduce_mean(tf.reduce_sum(oh * at, -1) * (-tf.pow(1 - pt, 2.0) * tf.math.log(pt)))
    m = keras.Model([e, r], out); m.compile(keras.optimizers.Adam(1e-3), loss=fl)
    return m, keras.Model([e, r], logits)

def train(seed, Xtr, ytr, Rtr, Xva, yva, Rva, width=0.15, conv2d=False, epochs=(35, 7, 20, 6)):
    import tensorflow as tf; from tensorflow import keras
    keras.backend.clear_session(); keras.utils.set_random_seed(seed)
    fmt = (lambda X: X[:, :, None, None]) if conv2d else (lambda X: X[:, :, None])
    present = [c for c in (1, 2) if (yva == c).any()]
    class Select(keras.callbacks.Callback):
        def __init__(s, pat): super().__init__(); s.pat = pat; s.best = -1; s.w = None; s.bad = 0
        def on_epoch_end(s, ep, logs=None):
            pr = s.model.predict([fmt(Xva), Rva], batch_size=2048, verbose=0).argmax(1)
            sc = float(np.mean([f1_score(yva, pr, labels=[c], average=None, zero_division=0)[0] for c in present]))
            if sc > s.best: s.best, s.w, s.bad = sc, s.model.get_weights(), 0
            else: s.bad += 1
            if s.bad >= s.pat: s.model.stop_training = True
        def on_train_end(s, logs=None): s.model.set_weights(s.w)
    counts = np.array([max(1, (ytr == i).sum()) for i in range(NC)], np.float64)
    sq = 1 / np.sqrt(counts); alpha = (sq / sq.sum()).tolist(); inv = 1 / counts; inv /= inv.min(); cw = {i: float(inv[i]) for i in range(NC)}
    p = np.random.permutation(len(Xtr)); Xa, ya, Ra = Xtr[p], ytr[p], Rtr[p]
    m, lm = build(width, conv2d, alpha)
    vdata = ([fmt(Xva), Rva], yva)
    cbs = lambda pat: [Select(pat), keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)]
    t0 = time.time()
    m.fit([fmt(Xa), Ra], ya, epochs=epochs[0], batch_size=256, validation_data=vdata, callbacks=cbs(epochs[1]), verbose=0, class_weight=cw)
    m.optimizer.learning_rate.assign(2e-4)
    m.fit([fmt(Xtr), Rtr], ytr, epochs=epochs[2], batch_size=128, validation_data=vdata, callbacks=cbs(epochs[3]), verbose=0, class_weight=cw)
    return m, lm, fmt, time.time() - t0

# ---------------- calibration + adaptation ----------------
def log_softmax(z): z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))

def fit_bcts(Z, y):
    """Bias-corrected temperature scaling: minimise NLL of softmax(Z/T + b) on labelled validation beats."""
    def nll(p):
        T = np.exp(p[0]); b = np.concatenate([[0.0], p[1:]]); return -log_softmax(Z / T + b)[np.arange(len(y)), y].mean()
    r = minimize(nll, np.zeros(NC), method='L-BFGS-B')
    return float(np.exp(r.x[0])), np.concatenate([[0.0], r.x[1:]])

def calibrated(Z, T, b): return np.exp(log_softmax(Z / T + b))

def adapt_stream(Q, pi_s, lam, floor=1e-4):
    """Causal online EM prior correction for one record (beats in time order). Q: calibrated posteriors under pi_s."""
    pi = pi_s.copy(); out = np.empty_like(Q); w = Q / pi_s
    for t in range(len(Q)):
        p = w[t] * pi; p /= p.sum(); out[t] = p
        pi = (1 - lam) * pi + lam * p; pi = np.maximum(pi, floor); pi /= pi.sum()
    return out

def adapt_offline(Q, pi_s, iters=100, floor=1e-4):
    pi = pi_s.copy(); w = Q / pi_s
    for _ in range(iters):
        P = w * pi; P /= P.sum(1, keepdims=True); new = np.maximum(P.mean(0), floor); new /= new.sum()
        if np.abs(new - pi).max() < 1e-7: pi = new; break
        pi = new
    P = w * pi; return P / P.sum(1, keepdims=True)

def adapt_oracle(Q, pi_s, y, floor=1e-4):
    pi = np.maximum(np.bincount(y[y >= 0], minlength=NC) / max(1, (y >= 0).sum()), floor); pi /= pi.sum()
    P = Q / pi_s * pi; return P / P.sum(1, keepdims=True)

def adapt_stream_prior(Q, pi_s, lam, floor=1e-4):
    """Returns pi_{t-1} for every beat t (the prior available BEFORE beat t), from the causal online EM."""
    pi = pi_s.copy(); out = np.empty_like(Q); w = Q / pi_s
    for t in range(len(Q)):
        out[t] = pi; p = w[t] * pi; p /= p.sum()
        pi = (1 - lam) * pi + lam * p; pi = np.maximum(pi, floor); pi /= pi.sum()
    return out

def offline_prior(Q, pi_s, iters=100, floor=1e-4):
    pi = pi_s.copy(); w = Q / pi_s
    for _ in range(iters):
        P = w * pi; P /= P.sum(1, keepdims=True); new = np.maximum(P.mean(0), floor); new /= new.sum()
        if np.abs(new - pi).max() < 1e-7: return new
        pi = new
    return pi

def apply_method(method, Z, y, rid, T, b, pi_s, lam=None):
    if method in ('M2c', 'M3c', 'M4c'):
        S = np.exp(log_softmax(Z)); Q = calibrated(Z, T, b); P = np.empty_like(S)
        for rec in dict.fromkeys(rid):
            k = np.where(rid == rec)[0]
            if method == 'M2c': pi = adapt_stream_prior(Q[k], pi_s, lam)
            elif method == 'M3c': pi = offline_prior(Q[k], pi_s)[None, :]
            else:
                yy = y[k]; pi = np.maximum(np.bincount(yy[yy >= 0], minlength=NC) / max(1, (yy >= 0).sum()), 1e-4); pi = (pi / pi.sum())[None, :]
            p = S[k] * pi / pi_s; P[k] = p / p.sum(1, keepdims=True)
        return P
    return _apply_method(method, Z, y, rid, T, b, pi_s, lam)

def _apply_method(method, Z, y, rid, T, b, pi_s, lam=None):
    """Returns posterior matrix for all beats, processing each record separately in stored (time) order."""
    if method == 'M0': return np.exp(log_softmax(Z))
    Q = calibrated(Z, T, b)
    if method == 'M1': return Q
    P = np.empty_like(Q)
    for rec in dict.fromkeys(rid):
        k = np.where(rid == rec)[0]
        P[k] = adapt_stream(Q[k], pi_s, lam) if method == 'M2' else adapt_offline(Q[k], pi_s) if method == 'M3' else adapt_oracle(Q[k], pi_s, y[k])
    return P

def metrics(y, pr, P=None):
    m = y >= 0; y, pr = y[m], pr[m]; res = {}
    for c in range(NC):
        tp = int(((pr == c) & (y == c)).sum()); n = int((y == c).sum()); npred = int((pr == c).sum())
        se = tp / max(1, n); pp = tp / max(1, npred)
        res[NAMES[c]] = dict(n=n, se=round(se, 4), ppv=round(pp, 4), f1=round(2*se*pp / max(1e-9, se + pp), 4))
    res['acc'] = round(float((pr == y).mean()), 4); res['kappa'] = round(float(cohen_kappa_score(y, pr)), 4)
    res['score'] = round((res['S']['f1'] + res['V']['f1']) / 2, 4)
    res['macroF1'] = round(float(np.mean([res[c]['f1'] for c in NAMES])), 4)
    return res
