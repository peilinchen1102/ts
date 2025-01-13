import torch
import numpy as np
from dataloaders.catalog import DATASET_DICT
from ECG_JEPA.ecg_data import waves_from_config, ECGDataset

DATA_DIR_DICT = {
    'CPSC':  '../high_modality/ecg/physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018',
    'ptbxl': '../high_modality/ecg/WFDB_PTBXL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/',
    'ChapmanShaoxing': '../local_ecg/ecg/WFDB_ChapmanShaoxing',
    'Ga':  '../high_modality/ecg/WFDB_Ga',
}

def get_dataset_transformer(dataset_name: str, fixed=True, window_size=10, overlap=5):
    base_root = '../high_modality_local' if dataset_name == 'ptbxl' else '../local_ecg'
    dataset = DATASET_DICT[dataset_name]
    train = dataset(base_root=base_root, fixed=fixed, window_size=window_size, overlap=overlap, train=True, download=False, dataset_name=dataset_name)
    test = dataset(base_root=base_root, fixed=fixed, window_size=window_size, overlap=overlap, train=False, download=False, dataset_name=dataset_name)
    return train, test

def get_dataset_jepa(dataset_name: str, reload=False):
    config = {
        'data_dir':  DATA_DIR_DICT[dataset_name],
        'dataset': dataset_name,
        'task': 'multiclass'
    }
    if reload:
        waves_train, waves_test, labels_train, labels_test = [np.load(f"{dataset_name}_{file}.npy") for file in ['waves_train', 'waves_test', 'labels_train', 'labels_test']]
    else:
        waves_train, waves_test, labels_train, labels_test = waves_from_config(config,reduced_lead=True)

    train = ECGDataset(waves_train, labels_train)
    test = ECGDataset(waves_test, labels_test)
    return train, test

