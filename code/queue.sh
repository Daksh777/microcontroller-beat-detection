export TF_ENABLE_ONEDNN_OPTS=0
cd /home/claude/work/v3
F="grep -E ^(cv|v2|causal|split|incart)"
for s in 42 1; do for k in 0 1 2 3; do python3 run_cv.py $k $s 2>&1 | $F; done; done
echo CV_DONE
for s in 42 1 2 3 4 5 6 7 8 9; do python3 run_final.py v2 $s 2>&1 | $F; done
echo V2_DONE
python3 -c "import sys; sys.path.insert(0,'.'); from lib import *; d=cached('c_incart', lambda: load_causal(sorted(os.path.basename(p)[:-4] for p in __import__('glob').glob(INCART+'/I*.hea')), 'incart')); print('incart', len(d['y']), np.bincount(d['y'][d['y']>=0]))" 2>&1 | $F
for s in 42 1 2 3 4 5 6 7 8 9; do python3 run_final.py causal $s 2>&1 | $F; done
echo CAUSAL_DONE
for i in $(seq 0 19); do python3 run_split.py $i 2>&1 | $F; done
echo ALL_DONE
