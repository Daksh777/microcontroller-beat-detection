# Supraventricular beat detection on microcontrollers: sources of error and patient calibration

Code, analysis plan and results for the paper *"Supraventricular Beat Detection on Microcontrollers: Sources of Error and Patient Calibration"*.

A 924-parameter dual-branch CNN (1-D CNN at 0.15x width plus a causal RR-feature branch) classifies heartbeats on MIT-BIH (AAMI EC57 classes N/S/V/F, inter-patient split: DS1 train, DS2 test), with INCART as external validation. It is trained with focal loss and class weights in two phases, with the checkpoint chosen by mean(F1_S, F1_V). The full-integer INT8 TFLite model is 9.5 KB and needs a 5.9 KB TensorFlow Lite Micro arena. Two front ends are compared: `v2` (offline: zero-phase filtering, 50 Hz notch, annotated R-peaks) and `causal` (real-time: one-pass Butterworth 0.5-45 Hz, 60 Hz notch, causal Pan-Tompkins detector, 150 ms matching). S-class sensitivity is very unstable, and patient class-prior shift explains much of the error. A pre-specified label-free streaming-EM prior correction (M2) failed. A few-label prior from the first 2 minutes (M5) helped in 17 of 20 random splits.

## Headline numbers (all from `results/`)

| Quantity | Value |
|---|---|
| Cohen's kappa, causal front end, DS2 (mean of 10 seeds, argmax) | 0.698 |
| Cohen's kappa, offline `v2`, DS2 | 0.715 |
| Cohen's kappa, causal, INCART | 0.788 |
| Parameters / INT8 model / TFLM arena | 924 / 9.5 KB / 5.9 KB |
| Causal detector: DS2 beats found | 99.8 % (49,616 / 49,712) |
| S sensitivity across 10 training seeds / 20 random patient splits | 0.25-0.81 / 0.00-0.85 |
| M5 (few-label prior) better than baseline | 17 of 20 splits |

## Limitations

- Not run on physical microcontroller hardware; latency and energy were not measured.
- The 5.9 KB arena was measured with a 64-bit host build of TFLite Micro (Python wrapper), not on a target MCU.
- The F class is essentially undetected.
- S-class results depend heavily on a single record (232 holds 1,382 of the 1,837 DS2 S beats).

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py          # MIT-BIH + INCART from PhysioNet into ./data (or set ECG_DATA_DIR)
```

`requirements.txt` gives version ranges; the exact package versions of the original runs were not recorded. The published results came from a cloud Linux environment with TensorFlow 2.x, with `TF_ENABLE_ONEDNN_OPTS=0` for the main runs. Bit-for-bit reproduction across TensorFlow versions and hardware is not guaranteed (the paper's oneDNN variance study says the same). `tflite-micro` is only needed for `code/quant_causal.py`. Data sources: [MIT-BIH](https://physionet.org/content/mitdb/1.0.0/), [INCART](https://physionet.org/content/incartdb/1.0.0/). Data are not redistributed here. Paced records 102, 104, 107 and 217 are excluded; DS1/DS2 (22 + 22 records) are defined in `code/lib.py`.

## Reproduce everything

```bash
bash scripts/run_all.sh
```

Every step skips outputs that already exist, so the script is resumable. All runs use `TF_ENABLE_ONEDNN_OPTS=0` except the explicit oneDNN-on variance study.

| Step | Command | Output |
|---|---|---|
| 1 | `prep.py`, `beattimes.py` | `cache/` (datasets, beat times for the "first k minutes" analysis) |
| 2 | `run_cv.py k seed`, k in 0..3, seed in {42, 1} | `results/cv_f{k}_s{seed}.npz` |
| 3 | `analyze_cv.py` | `results/cv_selection.json` (lambda = 1/800; adaptation not adopted) |
| 4 | `run_final.py v2 s` and `run_final.py causal s`, s in {42, 1..9}; then `TAG=dnn TF_ENABLE_ONEDNN_OPTS=1 run_final.py v2 s`, s in {42, 1..4} | `results/final_{v2,causal,v2dnn}_s*.npz`, `.keras` (git-ignored) |
| 5 | `run_split.py i`, i in 0..19 | `results/split_{i}.npz` |
| 6 | `quant_causal.py` | `results/causal_s*_int8.tflite`, `quant_causal.json`, logits npz |
| 7 | `analyze_prior.py cv`, `analyze_prior.py ds2`, `analyze_all.py` | `prior_cv.json`, `prior_ds2.json`, `v3_numbers.json` |
| 8 | `gen_tables.py`, `make_figs_v3.py` | `paper_out/` (LaTeX tables, `numbers_v3.json`, figures) |

Scripts live in `code/` and work from any directory (they switch to the repository root). The original run used two patch-up queues (splits 18-19 and v2 seed 42 rerun with oneDNN off, then the oneDNN-on runs); these are the same commands as above and are folded into steps 4-5.

Runtimes were not recorded.

## Script to paper map

| Script | Role |
|---|---|
| `lib.py` | data loading, both front ends, causal QRS detector, model, training, BCTS, EM prior correction, metrics |
| `analyze_all.py` -> `gen_tables.py` | all tables and flat numbers quoted in the text |
| `make_figs_v3.py` | variance, prior and confusion figures |
| `quant_causal.py` | INT8 conversion, TFLM arena (bisection), TFLM vs TFLite agreement on every DS2 beat |
| `figures/make_fig1.py` | Fig. 1 (pipeline and architecture); writes `fig1.png` in the current directory, needs only matplotlib |

## Methods and what is pre-specified

`analysis_plan/PREREGISTRATION_V3.txt` is the dated plan (2026-10-21, written before any v3 run), including amendment log A1-A4. It is unedited.

Methods: M0 argmax, M1 BCTS only, M2 streaming EM (lambda chosen on DS1 CV from {1/50..1/800}), M3 offline EM, M4 oracle prior. **Post hoc** (labelled so in the paper): M2c (cost-preserving), M5_k (few-label prior), and the ensembles.

## `results/`

- Committed summaries: `cv_selection.json`, `prior_cv.json`, `prior_ds2.json`, `quant_causal.json`, `v3_numbers.json`, and `c_ds{1,2}_stats.json` / `c_incart_stats.json` (causal detector statistics; copies of `cache/*_stats.json`).
- `causal_s{42,1,2,3,4}_int8.tflite`: the five quantised models (as in the paper, only seeds 42, 1-4 are quantised). All ten causal seeds (42, 1-9) were trained.

## Citation, license, contact

Cite the paper and the tagged release of this repository (`CITATION.cff` to be added). License: MIT (see `LICENSE`).
