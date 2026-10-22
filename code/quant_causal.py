"""Full-integer INT8 conversion of the causal-front-end models (logits output), TFLite + TFLite Micro checks."""
import sys, subprocess; sys.path.insert(0, '.'); from lib import *
import tensorflow as tf; from tensorflow import keras
SEEDS = [42, 1, 2, 3, 4]
D1, D2, DI = cached('c_ds1', None), cached('c_ds2', None), cached('c_incart', None)
f4 = lambda X: X[:, :, None, None].astype(np.float32)
lab = D1['y'] >= 0; tr = np.where(lab & ~np.isin(D1['rid'], VAL_RECS))[0]; rs = np.random.RandomState(0)
rep = np.concatenate([rs.choice(tr, 400, replace=False)] + [rs.choice(tr[D1['y'][tr] == c], 200, replace=False) for c in (1, 2, 3)])

def qin(x, d): s, z = d['quantization']; return np.clip(np.round(x / s + z), -128, 127).astype(np.int8)

def lite_logits(path, X, R, batch=4096):
    it = tf.lite.Interpreter(model_path=path); ins = it.get_input_details(); e = [i for i, d in enumerate(ins) if len(d['shape']) == 4][0]
    it.resize_tensor_input(ins[e]['index'], [batch, 180, 1, 1]); it.resize_tensor_input(ins[1-e]['index'], [batch, 5]); it.allocate_tensors()
    ins = it.get_input_details(); o = it.get_output_details()[0]; out = np.zeros((len(X), NC), np.float32)
    for k in range(0, len(X), batch):
        xe, xr = f4(X[k:k+batch]), R[k:k+batch].astype(np.float32); n = len(xe)
        if n < batch: xe = np.concatenate([xe, np.zeros((batch-n, 180, 1, 1), np.float32)]); xr = np.concatenate([xr, np.ones((batch-n, 5), np.float32)])
        it.set_tensor(ins[e]['index'], qin(xe, ins[e])); it.set_tensor(ins[1-e]['index'], qin(xr, ins[1-e])); it.invoke()
        s, z = o['quantization']; out[k:k+n] = (it.get_tensor(o['index'])[:n].astype(np.float32) - z) * s
    return out

def tflm_ok(path, n):
    code = f"from tflite_micro.python.tflite_micro import runtime\ntry:\n  runtime.Interpreter.from_file({path!r}, arena_size={n}); print('OK')\nexcept Exception: print('FAIL')"
    return 'OK' in subprocess.run([sys.executable, '-c', code], capture_output=True, text=True).stdout

rows = []
if os.path.exists('res/quant_causal.json'): sys.exit()
for s in SEEDS:
    m = keras.models.load_model(f'res/final_causal_s{s}.keras', compile=False)
    lm = keras.Model(m.inputs, m.get_layer('logits').output)
    def gen():
        for i in rep: yield {'ecg_input': f4(D1['X'][i:i+1]), 'rr_input': D1['R'][i:i+1].astype(np.float32)}
    c = tf.lite.TFLiteConverter.from_keras_model(lm); c.optimizations = [tf.lite.Optimize.DEFAULT]; c.representative_dataset = gen
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]; c.inference_input_type = tf.int8; c.inference_output_type = tf.int8
    path = f'res/causal_s{s}_int8.tflite'; open(path, 'wb').write(c.convert())
    Zt8, Zi8 = lite_logits(path, D2['X'], D2['R']), lite_logits(path, DI['X'], DI['R'])
    lo, hi = 256, 65536
    while hi - lo > 16:
        mid = (lo + hi) // 2; lo, hi = (lo, mid) if tflm_ok(path, mid) else (mid, hi)
    from tflite_micro.python.tflite_micro import runtime
    tm = runtime.Interpreter.from_file(path, arena_size=hi + 1024); te = 0 if len(tm.get_input_details(0)['shape']) == 4 else 1
    it = tf.lite.Interpreter(model_path=path); it.allocate_tensors(); li = it.get_input_details(); lo_ = it.get_output_details()[0]; e = [i for i, d in enumerate(li) if len(d['shape']) == 4][0]
    N = len(D2['y']); a = np.zeros(N, np.int64); bpred = np.zeros(N, np.int64); outq = lo_['quantization']; Ztm = np.zeros((N, NC), np.float32)
    for i in range(N):
        xe, xr = qin(f4(D2['X'][i:i+1]), li[e]), qin(D2['R'][i:i+1].astype(np.float32), li[1-e])
        tm.set_input(xe, te); tm.set_input(xr, 1 - te); tm.invoke(); o = tm.get_output(0)[0]; a[i] = o.argmax(); Ztm[i] = (o.astype(np.float32) - outq[1]) * outq[0]
        it.set_tensor(li[e]['index'], xe); it.set_tensor(li[1-e]['index'], xr); it.invoke(); bpred[i] = it.get_tensor(lo_['index'])[0].argmax()
    np.savez_compressed(f'res/causal_s{s}_int8_logits.npz', Zt8=Zt8, Zi8=Zi8, Ztm=Ztm)
    rows.append(dict(seed=s, bytes=os.path.getsize(path), arena=hi, tflm_agree=float((a == bpred).mean())))
    print(rows[-1], flush=True)
json.dump(rows, open('res/quant_causal.json', 'w'), indent=1)
