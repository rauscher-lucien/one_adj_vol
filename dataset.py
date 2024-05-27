import os
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from skimage import filters, exposure
import matplotlib.pyplot as plt
from PIL import Image

from utils import *


class TwoVolumeDataset(Dataset):
    def __init__(self, input_volume_path, target_volume_path, transform=None):
        self.input_volume_path = input_volume_path
        self.target_volume_path = target_volume_path
        self.transform = transform
        self.preloaded_data = {}  # To store preloaded data
        self.input_slices, self.target_slices = self.preload_volumes(input_volume_path, target_volume_path)

    def preload_volumes(self, input_volume_path, target_volume_path):
        # Preload input volume
        input_volume = tifffile.imread(input_volume_path)
        self.preloaded_data['input'] = input_volume
        num_slices_input = input_volume.shape[0]

        # Preload target volume
        target_volume = tifffile.imread(target_volume_path)
        self.preloaded_data['target'] = target_volume
        num_slices_target = target_volume.shape[0]

        # Ensure both volumes have the same number of slices
        if num_slices_input != num_slices_target:
            raise ValueError("Input and target volumes must have the same number of slices")

        input_slices = [(input_volume_path, i) for i in range(num_slices_input)]
        target_slices = [(target_volume_path, i) for i in range(num_slices_target)]

        return input_slices, target_slices

    def __len__(self):
        return len(self.input_slices)

    def __getitem__(self, index):
        input_path, input_slice_index = self.input_slices[index]
        target_path, target_slice_index = self.target_slices[index]

        # Access preloaded data instead of reading from file
        input_slice = self.preloaded_data['input'][input_slice_index][..., np.newaxis]
        target_slice = self.preloaded_data['target'][target_slice_index][..., np.newaxis]

        if self.transform:
            input_slice, target_slice = self.transform((input_slice, target_slice))

        return input_slice, target_slice
    


class N2NDataset(Dataset):
    def __init__(self, root_folder_path, num_volumes=None, transform=None):
        self.root_folder_path = root_folder_path
        self.num_volumes = num_volumes
        self.transform = transform
        self.preloaded_data = self.preload_volumes(root_folder_path)
        self.pairs = self.create_pairs()

    def preload_volumes(self, root_folder_path):
        preloaded_data = {}
        for subdir, _, files in os.walk(root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith('.tiff')])
            if self.num_volumes:
                sorted_files = sorted_files[:self.num_volumes]
                print(sorted_files)
            for f in sorted_files:
                full_path = os.path.join(subdir, f)
                volume = tifffile.imread(full_path)
                preloaded_data[full_path] = volume
        return preloaded_data

    def create_pairs(self):
        pairs = []
        volume_paths = sorted(self.preloaded_data.keys())
        num_volumes = len(volume_paths)

        if num_volumes < 2:
            raise ValueError("There must be at least two volumes in the folder")

        for i in range(num_volumes - 1):
            input_volume_path = volume_paths[i]
            target_volume_path = volume_paths[i + 1]
            num_slices = self.preloaded_data[input_volume_path].shape[0]
            for slice_index in range(num_slices):
                pairs.append((input_volume_path, target_volume_path, slice_index))
        
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        input_volume_path, target_volume_path, slice_index = self.pairs[index]

        input_slice = self.preloaded_data[input_volume_path][slice_index][..., np.newaxis]
        target_slice = self.preloaded_data[target_volume_path][slice_index][..., np.newaxis]

        if self.transform:
            input_slice, target_slice = self.transform((input_slice, target_slice))

        return input_slice, target_slice





class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, transform=None):
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.preloaded_data = {}  # To store preloaded data
        self.pairs = self.preload_and_make_pairs(root_folder_path)

    def preload_and_make_pairs(self, root_folder_path):
        pairs = []
        for subdir, _, files in os.walk(root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith(('.tif', '.tiff'))])
            for f in sorted_files:
                full_path = os.path.join(subdir, f)
                volume = tifffile.imread(full_path)
                self.preloaded_data[full_path] = volume  # Preload data here
                num_slices = volume.shape[0]
                for i in range(num_slices - 1):  # Ensure there is a next slice
                    input_slice_index = i
                    target_slice_index = i + 1
                    pairs.append((full_path, input_slice_index, target_slice_index))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        file_path, input_slice_index, target_slice_index = self.pairs[index]
        
        # Access preloaded data instead of reading from file
        input_slice = self.preloaded_data[file_path][input_slice_index][..., np.newaxis]
        target_slice = self.preloaded_data[file_path][target_slice_index][..., np.newaxis]

        if self.transform:
            input_slice = self.transform((input_slice, target_slice))

        return input_slice