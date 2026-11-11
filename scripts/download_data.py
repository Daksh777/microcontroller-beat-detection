"""Download MIT-BIH and INCART from PhysioNet into $ECG_DATA_DIR (default ./data). Data are not committed."""
import os, wfdb
root = os.path.abspath(os.environ.get('ECG_DATA_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')))
wfdb.dl_database('mitdb', os.path.join(root, 'mit-bih-arrhythmia-database-1.0.0'))
wfdb.dl_database('incartdb', os.path.join(root, 'st-petersburg-incart-12-lead-arrhythmia-database-1.0.0', 'files'))
print('data in', root)
