import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from scipy.io import loadmat
from scipy.signal import decimate, resample
from torchvision.datasets.utils import download_and_extract_archive

class Input1dSpec(object):
    '''Defines the specs for 1d inputs.'''

    input_type = '1d'

    def __init__(self, seq_len: int, segment_size: int, in_channels: int):
        self.seq_len = seq_len
        self.segment_size = segment_size
        self.in_channels = in_channels

class ptbxl(data.Dataset):
    '''Transform and return PTB-XL EKG dataset. Each example contains 5000 = 10 seconds * 500Hz 12-channel measurements. All datasets are by themselves 500Hz or resampled 
       in 500 Hz.
    '''
    # Dataset information.
    ECG_RESOURCES = {
        'Ga': 'https://pipelineapi.org:9555/api/download/physionettraining/WFDB_Ga.tar.gz',
        'CPSC': 'https://pipelineapi.org:9555/api/download/physionettraining/WFDB_CPSC2018.tar.gz',
        'Chapman-Shaoxing': 'https://physionet.org/files/ecg-arrhythmia/1.0.0',
        'ptbxl': 'https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip'
    }

    MEASUREMENTS_PER_EXAMPLE = 5000  # Measurements used
    REC_FREQ = 500  #Recording frequency here are all set to 500Hz
    SEGMENT_SIZE = 25
    IN_CHANNELS = 12  # Multiple sensor readings from different parts of the body
    REC_DIMS = (IN_CHANNELS, MEASUREMENTS_PER_EXAMPLE)

    LABEL_FRACS = {'small': 8, 'medium': 64, 'large': 256, 'full': np.inf}

    CLASSES = ['normal', 'cd', 'mi', 'sttc', 'other', 'afib', 'hyp']  #7 unified classes
    NUM_CLASSES = len(CLASSES)
    DATASET_PATH_MAP = {
        'CPSC': 'WFDB_CPSC2018',
        'Chapman-Shaoxing': 'WFDB_ChapmanShaoxing',
        'Ga': 'WFDB_Ga',
        'ptbxl': 'WFDB_PTBXL'
    }  # Dict to map datasets with different folder names

    def __init__(
        self,
        base_root: str,
        download: bool = False,
        train: bool = True,
        dataset_name: str = 'ptbxl',
        finetune_size: str = None
    ) -> None:
        super().__init__()
        self.base_root = base_root
        self.root = os.path.join(base_root, 'ecg/', self.DATASET_PATH_MAP[dataset_name])
        self.csv = os.path.join(base_root, 'ecg/')
        self.mode = 'train' if train else 'val'
        self.finetune_size = 0 if finetune_size is None else ptbxl.LABEL_FRACS[finetune_size]
        self.ds_name = dataset_name
        if download:
            self.download_dataset()

        if not self._is_downloaded():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        self.subject_data = self.load_data()

    def _is_downloaded(self):
        print(self.root)
        return (os.path.exists(self.root))

    def download_dataset(self):
        '''Download the dataset if not exists already'''

        if self._is_downloaded():
            return

        # download and extract files
        print('Downloading and Extracting...')

        filename = self.ECG_RESOURCES[self.ds_name].rpartition('/')[2]
        download_and_extract_archive(self.ECG_RESOURCES[self.ds_name], download_root=self.root, filename=filename)

        print('Done!')

    # Prefix is defined by the data source provider Physionet, to differentiate patient and their data across different datasets,
    # by accessing different prefixes we can easily filter out the data we need using one base dataset class
    def load_data(self):
        data = pd.read_csv(self.csv + f'/{self.ds_name}_splits.csv')
        data = data[data.split == self.mode].reset_index(drop=True)
        df = data.copy()
        if self.mode == 'train' and self.finetune_size > 0:
            # Get counts for every label
            unique_counts = df.loc[:, 'label'].value_counts()
            train_df = pd.DataFrame(columns=df.columns)

            for label, count in unique_counts.items():
                # Get 'finetune_size' labels for each class
                # if insufficient examples, use all examples from that class
                num_sample = min(self.finetune_size, count)
                train_rows = df.loc[df.loc[:, 'label'] == label].sample(num_sample, random_state=0)
                train_df = train_df.append(train_rows)

            df = train_df

        return df

    def load_measurements(self, index):
        i = index
        recording = ptbxl._read_recording(self.root, self.subject_data.iloc[i]["patient"], self.REC_DIMS)
        label = self.subject_data.iloc[i]['label']
        return recording, label

    def __getitem__(self, index):

        measurements, label = self.load_measurements(index)
        return (index, measurements, label)

    def __len__(self):
        return self.subject_data.shape[0]

    @staticmethod
    def num_classes():
        return ptbxl.NUM_CLASSES

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input1dSpec(
                seq_len=int(ptbxl.MEASUREMENTS_PER_EXAMPLE), segment_size=ptbxl.SEGMENT_SIZE, in_channels=ptbxl.IN_CHANNELS
            ),
        ]

    @staticmethod
    def _read_recording(path: str, id: str, rdim: Tuple):
        file_name = path + '/' + id
        _, rL = rdim
        recording = ptbxl._process_recording(file_name)
        C, _ = recording.shape
        recording = recording[:, :rL].view(C, -1, rL).squeeze(1).transpose(0, 1)  #exhaustive crop
        return recording.contiguous().float()

    @staticmethod
    def _process_recording(file_name: str):
        recording = loadmat(f"{file_name}.mat")['val'].astype(float)

        # Standardize sampling rate, sampling rate across different datasets here are already normalized by data provider Physionet, but we added this logic in case users want to use their own new datasets
        sampling_rate = ptbxl._get_sampling_rate(file_name)

        if sampling_rate > ptbxl.REC_FREQ:
            recording = np.copy(decimate(recording, int(sampling_rate / ptbxl.REC_FREQ)))
        elif sampling_rate < ptbxl.REC_FREQ:
            recording = np.copy(resample(recording, int(recording.shape[-1] * (ptbxl.REC_FREQ / sampling_rate)), axis=1))

        return torch.from_numpy(ptbxl._normalize(recording))

    @staticmethod
    def _normalize(x: np.ndarray):
        return x / (np.max(x) - np.min(x))

    @staticmethod
    def _get_sampling_rate(file_name: str):
        with open(f"{file_name}.hea", 'r') as f:
            return int(f.readline().split(None, 3)[2])