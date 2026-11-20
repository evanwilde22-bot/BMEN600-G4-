'''
Dataset for training/testing MNet on the Soft-tissue-Sarcoma (STS) NIfTI
dataset.

This mirrors datasets/brains18.py but adds a binarization step for the
segmentation masks, since the STS RTSTRUCT-derived masks are stored with
foreground voxels at 255 (not 1). Without this remap, CrossEntropyLoss
would receive out-of-range class labels during training/evaluation.

The legacy training path trims background before resizing, while legacy
testing resizes the full volume. The opt-in consistent_preprocessing mode
uses the full volume and the same normalization for training and testing.

NOTE on augmentation: with the tumor-centric crop removed, training lost
its only source of randomness -- every epoch saw the exact same
deterministic volume for each case, which combined with a small dataset
(~40 training cases) and training from scratch (no pretrained weights)
caused severe overfitting (train Dice ~0.99, validation Dice ~0.01).
Training now applies random axis flips and mild random intensity jitter.
These augmentations don't change the physical scale/size of the tumor
within the volume, so they don't reintroduce the train/test scale
mismatch the crop removal was meant to fix.

Expected file list format (train.txt / val.txt), one case per line:
    <relative image path> <relative mask path>

Example:
    STS_001/RTstruct_T1/image.nii.gz STS_001/RTstruct_T1/mask_GTV_Mass.nii.gz
'''

import os

import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage


def normalize_consistent_volume(volume):
    volume = np.asarray(volume, dtype=np.float32)
    if not np.isfinite(volume).all():
        raise ValueError("Image contains non-finite intensities")
    pixels = volume[volume != 0]
    if not pixels.size or pixels.std() == 0:
        raise ValueError("Image has no intensity variation")
    normalized = (volume - pixels.mean()) / pixels.std()
    normalized[volume == 0] = 0
    return normalized


def resize_volume(data, input_size):
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {data.shape}")
    scale = [target / actual for target, actual in zip(input_size, data.shape)]
    return ndimage.zoom(data, scale, order=0)


class SarcomaDataset(Dataset):

    def __init__(self, root_dir, img_list, sets):
        with open(img_list, 'r') as f:
            self.img_list = [line.strip() for line in f if line.strip()]
        print("Processing {} datas".format(len(self.img_list)))
        self.root_dir = root_dir
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.phase = sets.phase
        self.consistent_preprocessing = getattr(sets, 'consistent_preprocessing', False)

    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x])
        new_data = new_data.astype("float32")

        return new_data

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        if self.phase == "train":
            # read image and labels
            ith_info = self.img_list[idx].split(" ")
            img_name = os.path.join(self.root_dir, ith_info[0])
            label_name = os.path.join(self.root_dir, ith_info[1])
            assert os.path.isfile(img_name)
            assert os.path.isfile(label_name)
            img = nibabel.load(img_name)
            assert img is not None
            mask = nibabel.load(label_name)
            assert mask is not None
            if self.consistent_preprocessing and (
                    img.shape != mask.shape or not np.allclose(img.affine, mask.affine, atol=1e-3)):
                raise ValueError(f"Image and mask grids differ: {img_name}, {label_name}")

            # data processing
            img_array, mask_array = self.__training_data_process__(img, mask)

            # 2 tensor array
            img_array = self.__nii2tensorarray__(img_array)
            mask_array = self.__nii2tensorarray__(mask_array)

            assert img_array.shape == mask_array.shape, \
                "img shape:{} is not equal to mask shape:{}".format(img_array.shape, mask_array.shape)
            return img_array, mask_array

        elif self.phase == "test":
            # read image
            ith_info = self.img_list[idx].split(" ")
            img_name = os.path.join(self.root_dir, ith_info[0])
            print(img_name)
            assert os.path.isfile(img_name)
            img = nibabel.load(img_name)
            assert img is not None

            # data processing
            img_array = self.__testing_data_process__(img)

            # 2 tensor array
            img_array = self.__nii2tensorarray__(img_array)

            return img_array

    def __binarize_label__(self, label):
        """
        Remap STS RTSTRUCT-derived masks (foreground stored as 255) to
        standard integer class labels (0 = background, 1 = foreground).
        Safe to call even if masks are already 0/1.
        """
        label = np.where(label > 0, 1, 0).astype("float32")
        return label

    def __drop_invalid_range__(self, volume, label=None):
        """
        Cut off the invalid area
        """
        zero_value = volume[0, 0, 0]
        non_zeros_idx = np.where(volume != zero_value)

        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)

        if label is not None:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w], label[min_z:max_z, min_h:max_h, min_w:max_w]
        else:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w]

    def __random_flip__(self, data, label):
        """
        Randomly flip the volume (and matching label) along each spatial
        axis independently with 50% probability each. Flips don't change
        tumor scale/size, only orientation, so they're safe augmentation
        that doesn't reintroduce the train/test scale mismatch.
        """
        for axis in range(3):
            if np.random.rand() < 0.5:
                data = np.flip(data, axis=axis)
                label = np.flip(label, axis=axis)
        # copy needed since np.flip returns a view with negative strides,
        # which torch.from_numpy can't handle directly
        return np.ascontiguousarray(data), np.ascontiguousarray(label)

    def __random_intensity_jitter__(self, data, scale_range=0.1, shift_range=0.1):
        """
        Apply a mild random per-volume intensity scale and shift. Helps
        reduce overfitting to the exact intensity distribution of the
        small training set, without altering anatomical geometry.
        """
        scale = 1.0 + np.random.uniform(-scale_range, scale_range)
        shift = np.random.uniform(-shift_range, shift_range)
        return data * scale + shift

    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """

        pixels = volume[volume > 0]
        mean = pixels.mean()
        std = pixels.std()
        out = (volume - mean) / std
        out_random = np.random.normal(0, 1, size=volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out

    def __resize_data__(self, data):
        """
        Resize the data to the input size
        """
        return resize_volume(data, (self.input_D, self.input_H, self.input_W))

    def __training_data_process__(self, data, label):
        # crop data according net input size
        data = np.asanyarray(data.dataobj)
        label = np.asanyarray(label.dataobj)

        # remap STS mask foreground (255) to class label 1
        label = self.__binarize_label__(label)

        # drop out the invalid (empty background/air) range only -- no
        # tumor-centric crop, so train-time scale matches test-time scale
        if not self.consistent_preprocessing:
            data, label = self.__drop_invalid_range__(data, label)

        # random augmentation (train only): flips + mild intensity jitter.
        # Neither changes tumor scale, so train/test scale consistency is
        # preserved while still giving the model varied training samples.
        data, label = self.__random_flip__(data, label)
        if not self.consistent_preprocessing:
            data = self.__random_intensity_jitter__(data)

        # resize data
        data = self.__resize_data__(data)
        label = self.__resize_data__(label)

        # normalization datas
        if self.consistent_preprocessing:
            data = self.__random_intensity_jitter__(normalize_consistent_volume(data))
        else:
            data = self.__itensity_normalize_one_volume__(data)

        return data, label

    def __testing_data_process__(self, data):
        # crop data according net input size
        data = np.asanyarray(data.dataobj)

        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        if self.consistent_preprocessing:
            data = normalize_consistent_volume(data)
        else:
            data = self.__itensity_normalize_one_volume__(data)

        return data
