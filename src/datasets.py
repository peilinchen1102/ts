import torch
import numpy as np
from sklearn.utils import shuffle
from load_datasets.catalog import DATASET_DICT
from ECG_JEPA.ecg_data import waves_from_config, ECGDataset
from ECG_JEPA.augmentation import *

DATA_DIR_DICT = {
    'CPSC': '../local_ecg/ecg/WFDB_CPSC2018',
    'ptbxl': '../high_modality_local/ecg/ptbxl/WFDB',
    'ChapmanShaoxing': '../local_ecg/ecg/WFDB_ChapmanShaoxing',
    'Ga':  '../high_modality/ecg/WFDB_Ga',
}

def get_dataset_transformer(dataset_name: str, fixed=True, window_size=10, overlap=5):
    dataset_name = dataset_name[0]
    base_root = '../high_modality_local' if dataset_name == 'ptbxl' else '../local_ecg'
    dataset = DATASET_DICT[dataset_name]
    train = dataset(base_root=base_root, fixed=fixed, window_size=window_size, overlap=overlap, train=True, download=False, dataset_name=dataset_name)
    test = dataset(base_root=base_root, fixed=fixed, window_size=window_size, overlap=overlap, train=False, download=False, dataset_name=dataset_name)
    return train, test

def get_dataset_jepa(dataset_names, reload=False):
    aug = {
        'transforms': [{'highpass_filter': {'fs': 250, 'cutoff': 0.67}}, {'lowpass_filter': {'fs': 250, 'cutoff': 40}}, {'standardize': {}}]
    }

    transforms = Compose(get_transforms_from_config(aug["transforms"]) + [ToTensor()])

    if len(dataset_names) == 1:
        dataset_name = dataset_names[0]
        config = {
            'data_dir':  DATA_DIR_DICT[dataset_name],
            'dataset': dataset_name,
            'task': 'multiclass'
        }
        if not reload:
            waves_train, waves_test, labels_train, labels_test = [np.load(f"np_datasets/{dataset_name}_{file}.npy") for file in ['waves_train', 'waves_test', 'labels_train', 'labels_test']]
        else:
            waves_train, waves_test, labels_train, labels_test = waves_from_config(config,reduced_lead=True)

        train = ECGDataset(waves_train, labels_train, transforms)
        test = ECGDataset(waves_test, labels_test, transforms)

    else:
        all_waves_train, all_waves_test = [], []
        all_labels_train, all_labels_test = [], [] 

        for dataset_name in dataset_names:
            config = {
                'data_dir':  DATA_DIR_DICT[dataset_name],
                'dataset': dataset_name,
                'task': 'multiclass'
            }
            if not reload:
                waves_train, waves_test, labels_train, labels_test = [np.load(f"np_datasets/{dataset_name}_{file}.npy") for file in ['waves_train', 'waves_test', 'labels_train', 'labels_test']]
            else:
                waves_train, waves_test, labels_train, labels_test = waves_from_config(config,reduced_lead=True)

            all_waves_train.append(waves_train)
            all_waves_test.append(waves_test)
            all_labels_train.append(labels_train)
            all_labels_test.append(labels_test)

        all_waves_train = np.concatenate(all_waves_train, axis=0)
        all_labels_train = np.concatenate(all_labels_train, axis=0)
        all_waves_train, all_labels_train = shuffle(all_waves_train, all_labels_train, random_state=42)

        all_waves_test = np.concatenate(all_waves_test, axis=0)
        all_labels_test = np.concatenate(all_labels_test, axis=0)

        train = ECGDataset(all_waves_train, all_labels_train, transforms)
        test = ECGDataset(all_waves_test, all_labels_test, transforms)
  
    return train, test

