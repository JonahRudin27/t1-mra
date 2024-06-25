from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from misc_utils import get_matched_ids
import os


class T1w2MraDataset(Dataset):
    '''
    Dataset class for T1w to MRA data

    Parameters
    ----------
    mri_dir : str
        Path to the directory containing the T1w MRI images
    mra_dir : str
        Path to the directory containing the MRA images
    slice_axis : int or string, optional
        Axis along which to slice the 3D MRI images. Default is "all",
        an int specifies axes as follows:
        0 = Sagittal (x-axis), 1 = Coronal (y-axis), 2 = Axial (z-axis)
    transform : callable, optional
        Transform to apply to the data
    split_char : str, optional
        Character to split the file names UID, default is "-"
    '''
    def __init__(self, mri_dir, mra_dir, transform, slice_axis=2,
                 split_char="-"):
        self.mri_dir = mri_dir
        self.mra_dir = mra_dir
        self.slice_axis = slice_axis
        self.transform = transform
        self.split_char = split_char

        self.mri_paths = [os.path.join(mri_dir, filename) for filename in
                          sorted(os.listdir(mri_dir))]
        self.mra_paths = [os.path.join(mra_dir, filename) for filename in
                          sorted(os.listdir(mra_dir))]

        self.id_list = self._get_id_list()

    def __len__(self):
        return self.id_list[-1].get("total_running_slices")

    def __getitem__(self, idx):
        length = len(self)
        file_idx = idx // self.id_list[0].get("total_running_slices")
        while True:
            slice_idx = self.id_list[file_idx].get("total_running_slices")
            slice_low_idx = self.id_list[file_idx - 1].get("total_running_slices") if file_idx > 0 else 0

            # Check if idx is within the current range
            if (slice_low_idx is None or idx >= slice_low_idx) and (idx <= slice_idx):
                mri_path = self.id_list[file_idx].get("mri_path")
                mra_path = self.id_list[file_idx].get("mra_path")
                my_mri = nib.load(mri_path).get_fdata()
                my_mra = nib.load(mra_path).get_fdata()
                slice_idx = idx - slice_low_idx
                mri_slice = my_mri[:, :, slice_idx]
                mra_slice = my_mra[:, :, slice_idx]

                mri_slice = self.transform(mri_slice)
                mra_slice = self.transform(mra_slice)

                return mri_slice, mra_slice
            # Go to next or previous file index
            elif (idx < slice_low_idx and idx >= 0):
                file_idx -= 1
            elif (idx >= slice_idx and idx < length):
                file_idx += 1
            else:
                raise IndexError("Index out of range")

    def _get_id_list(self):

        ids = get_matched_ids([self.mri_dir, self.mra_dir],
                              split_char=self.split_char)
        id_list = []
        slices = 0
        for i, id in enumerate(ids):
            matching_mri = [path for path in self.mri_paths if id in path]
            matching_mra = [path for path in self.mra_paths if id in path]

            if len(matching_mri) == 1 and len(matching_mra) == 1:
                slices += self._get_num_slices(matching_mri[0])
                id_list.append({"mri_path": matching_mri[0],
                                "mra_path": matching_mra[0],
                                "total_running_slices": slices})
            else:
                raise ValueError(f"ID {id} has {len(matching_mri)} MRI images "
                                 f"and {len(matching_mra)} MRA images. There "
                                 f"should be exactly one of each.")
        return id_list

    def _get_num_slices(self, filepath):
        '''
        Returns the number of slices for the MRI image input
        '''
        if self.slice_axis == "all":
            shape = nib.load(filepath).get_fdata().shape()
            return sum(shape)
        else:
            return nib.load(filepath).get_fdata().shape[self.slice_axis]

    def _get_shape(self):
        '''
        Returns the shape of the MRI images
        '''
        key = next(iter(self.id_dict))
        path = self.id_dict[key]["mri_path"]
        return nib.load(path).get_fdata().shape()
