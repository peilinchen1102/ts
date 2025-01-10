
from load_datasets import ChapmanShaoxing, CPSC, Ga, ptbxl

DATASET_DICT = {
    # ECG
    'ptbxl': ptbxl.ptbxl,
    'ChapmanShaoxing': ChapmanShaoxing.ChapmanShaoxing,
    'CPSC': CPSC.CPSC,
    'Ga': Ga.Ga,
}