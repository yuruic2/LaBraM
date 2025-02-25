import h5py
import bisect
from pathlib import Path
from typing import List
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from mef_tools.io import MefReader

import utils

list_path = List[Path]

# class SingleShockDataset(Dataset):
#     """Read single hdf5 file regardless of label, subject, and paradigm."""
#     def __init__(self, file_path: Path, window_size: int=200, stride_size: int=1, start_percentage: float=0, end_percentage: float=1):
#         '''
#         Extract datasets from file_path.

#         param Path file_path: the path of target data
#         param int window_size: the length of a single sample
#         param int stride_size: the interval between two adjacent samples
#         param float start_percentage: Index of percentage of the first sample of the dataset in the data file (inclusive)
#         param float end_percentage: Index of percentage of end of dataset sample in data file (not included)
#         '''
#         self.__file_path = file_path
#         self.__window_size = window_size
#         self.__stride_size = stride_size
#         self.__start_percentage = start_percentage
#         self.__end_percentage = end_percentage

#         self.__file = None
#         self.__length = None
#         self.__feature_size = None

#         self.__subjects = []
#         self.__global_idxes = []
#         self.__local_idxes = []
        
#         self.__init_dataset()

#     def __init_dataset(self) -> None:
#         self.__file = h5py.File(str(self.__file_path), 'r')
#         self.__subjects = [i for i in self.__file]

#         global_idx = 0
#         for subject in self.__subjects:
#             self.__global_idxes.append(global_idx) # the start index of the subject's sample in the dataset
#             subject_len = self.__file[subject]['eeg'].shape[1]
#             # total number of samples
#             total_sample_num = (subject_len-self.__window_size) // self.__stride_size + 1
#             # cut out part of samples
#             start_idx = int(total_sample_num * self.__start_percentage) * self.__stride_size 
#             end_idx = int(total_sample_num * self.__end_percentage - 1) * self.__stride_size

#             self.__local_idxes.append(start_idx)
#             global_idx += (end_idx - start_idx) // self.__stride_size + 1
#         self.__length = global_idx

#         self.__feature_size = [i for i in self.__file[self.__subjects[0]]['eeg'].shape]
#         self.__feature_size[1] = self.__window_size

#     @property
#     def feature_size(self):
#         return self.__feature_size

#     def __len__(self):
#         return self.__length

#     def __getitem__(self, idx: int):
#         subject_idx = bisect.bisect(self.__global_idxes, idx) - 1
#         item_start_idx = (idx - self.__global_idxes[subject_idx]) * self.__stride_size + self.__local_idxes[subject_idx]
#         return self.__file[self.__subjects[subject_idx]]['eeg'][:, item_start_idx:item_start_idx+self.__window_size]
    
#     def free(self) -> None: 
#         if self.__file:
#             self.__file.close()
#             self.__file = None
    
#     def get_ch_names(self):
#         return self.__file[self.__subjects[0]]['eeg'].attrs['chOrder']


################# YC - 2025/02/13: Prepare dataset from mefd file.
class SingleShockDataset(Dataset):
    """Read single hdf5 file regardless of label, subject, and paradigm."""
    def __init__(self, file_path: Path, window_size: int=200, stride_size: int=1, start_percentage: float=0, end_percentage: float=1):
        '''
        Extract datasets from file_path.

        param Path file_path: the path of target data
        param int window_size: the length of a single sample
        param int stride_size: the interval between two adjacent samples
        param float start_percentage: Index of percentage of the first sample of the dataset in the data file (inclusive)
        param float end_percentage: Index of percentage of end of dataset sample in data file (not included)
        '''
        self.__file_path = str(file_path)
        self.__idx_path = file_path.with_suffix(".csv")
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage

        self.__subjects = []
        self.__global_idxes = []
        self.__local_idxes = []

        self.__file = None
        self.__elec_list = None
        self.__channel_list = None
        self.__freq = None

        self.__length = None
        self.__feature_size = None

        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__file = MefReader(self.__file_path)
        self.__idx_list = pd.read_csv(self.__idx_path)

        self.__elec_list = self.__file.channels
        self.__channel_list = [utils.elec_to_chans(elec) for elec in self.__elec_list]
        self.__freq = self.__file.get_property('fsamp', self.__elec_list[0])
        
        global_idx = 0
        for _, row in self.__idx_list.iterrows():
            self.__global_idxes.append(global_idx) # the start index of the subject's sample in the dataset
            subject_len = (row['end'] - row['start']) / 1e6 * self.__freq
            # total number of samples
            total_sample_num = (subject_len-self.__window_size) // self.__stride_size + 1
            # cut out part of samples
            start_idx = int(total_sample_num * self.__start_percentage) * self.__stride_size 
            end_idx = int(total_sample_num * self.__end_percentage - 1) * self.__stride_size

            self.__local_idxes.append(start_idx)
            global_idx += (end_idx - start_idx) // self.__stride_size + 1
        self.__length = global_idx

        self.__feature_size = [len(self.__channel_list), self.__window_size]

    @property
    def feature_size(self):
        return self.__feature_size

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        subject_idx = bisect.bisect(self.__global_idxes, idx) - 1
        item_start_idx = (idx - self.__global_idxes[subject_idx]) * self.__stride_size + self.__local_idxes[subject_idx]

        subject_start = self.__idx_list.iloc[subject_idx]['start']
        item_start = int(subject_start + item_start_idx * 1e6 / self.__freq)
        item_end = int(subject_start + (item_start_idx + self.__window_size) * 1e6 / self.__freq)

        data = np.zeros((len(self.__channel_list), self.__window_size))
        for nch, elec in enumerate(self.__elec_list):
            data[nch, :] = self.__file.get_data(elec, item_start, item_end)

        # this line convert uV to mV and scale data by 10 to have range (-1, 1)
        return data / (10 * 1e3)

    def free(self) -> None: 
        if self.__file:
            self.__file.close()
            self.__file = None
    
    def get_ch_names(self):
        return self.__channel_list


class ShockDataset(Dataset):
    """integrate multiple hdf5 files"""
    def __init__(self, file_paths: list_path, window_size: int=200, stride_size: int=1, start_percentage: float=0, end_percentage: float=1):
        '''
        Arguments will be passed to SingleShockDataset. Refer to SingleShockDataset.
        '''
        ################# YC - 2025/02/21: Modify to gather mefd file in the folder.
        self.__file_paths = file_paths[0].glob('*.mefd')
        # self.__file_paths = file_paths
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage

        self.__datasets = []
        self.__length = None
        self.__feature_size = None

        self.__dataset_idxes = []
        
        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__datasets = [SingleShockDataset(file_path, self.__window_size, self.__stride_size, self.__start_percentage, self.__end_percentage) for file_path in self.__file_paths]
        
        # calculate the number of samples for each subdataset to form the integral indexes
        dataset_idx = 0
        for dataset in self.__datasets:
            self.__dataset_idxes.append(dataset_idx)
            dataset_idx += len(dataset)
        self.__length = dataset_idx

        self.__feature_size = self.__datasets[0].feature_size

    @property
    def feature_size(self):
        return self.__feature_size

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        dataset_idx = bisect.bisect(self.__dataset_idxes, idx) - 1
        item_idx = (idx - self.__dataset_idxes[dataset_idx])
        return self.__datasets[dataset_idx][item_idx]
    
    def free(self) -> None:
        for dataset in self.__datasets:
            dataset.free()
    
    def get_ch_names(self):
        return self.__datasets[0].get_ch_names()
