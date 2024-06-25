from torch.utils.data import Dataset
import nibabel as nib
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
    def __init__(self, mri_dir, mra_dir, slice_axis="all", transform=None,
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

        self.id_dict = self._get_id_dict()

    def __len__(self):
        samples = len(self._get_id_dict())

        key = next(iter(self.id_dict))
        path = self.id_dict[key]["mri_path"]
        slices = self._get_num_slices(path)

        return samples * slices

    def __getitem__(self, idx):


    def _get_id_dict(self):

        ids = get_matched_ids([self.mri_dir, self.mra_dir],
                              split_char=self.split_char)
        id_dict = {}

        for id in ids:
            matching_mri = [path for path in self.mri_paths if id in path]
            matching_mra = [path for path in self.mra_paths if id in path]
            if len(matching_mri) == 1 and len(matching_mra) == 1:
                id_dict[id] = {"mri_path": matching_mri[0],
                               "mra_path": matching_mra[0]}
            else:
                raise ValueError(f"ID {id} has {len(matching_mri)} MRI images "
                                 f"and {len(matching_mra)} MRA images. There "
                                 f"should be exactly one of each.")
        return id_dict

    def _get_num_slices(self, filepath):
        '''
        Returns the number of slices for the MRI image input
        '''
        key = next(iter(self.id_dict))
        path = self.id_dict[key]["mri_path"]
        if self.slice_axis == "all":
            shape = nib.load(path).get_fdata().shape()
            return sum(shape)
        else:
            return nib.load(path).get_fdata().shape[self.slice_axis]

    def _get_shape(self):
        '''
        Returns the shape of the MRI images
        '''
        key = next(iter(self.id_dict))
        path = self.id_dict[key]["mri_path"]
        return nib.load(path).get_fdata().shape()