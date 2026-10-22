cd /home/claude/work/v3
for i in 18 19; do TF_ENABLE_ONEDNN_OPTS=0 python3 run_split.py $i 2>&1 | grep -E "^split|Traceback"; done
TF_ENABLE_ONEDNN_OPTS=0 python3 run_final.py v2 42 2>&1 | grep -E "^v2|Traceback"
for s in 42 1 2 3 4; do TAG=dnn TF_ENABLE_ONEDNN_OPTS=1 python3 run_final.py v2 $s 2>&1 | grep -E "^v2|Traceback"; done
echo Q3_DONE
