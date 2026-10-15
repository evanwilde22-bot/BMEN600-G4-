'''
Dataset for training/testing MNet on the Soft-tissue-Sarcoma (STS) NIfTI
dataset.

This mirrors datasets/brains18.py but adds a binarization step for the
segmentation masks, since the STS RTSTRUCT-derived masks are stored with
foreground voxels at 255 (not 1). Without this remap, CrossEntropyLoss
would receive out-of-range class labels during training/evaluation.

NOTE on cropping: training intentionally does NOT use a random tumor-centric
crop (unlike the original brains18.py). Testing has no knowledge of where
the tumor is, so it only trims empty background/air (__drop_invalid_range__)
before resizing to the network input size. If training cropped tightly
around the tumor, the tumor would occupy a much larger fraction of the
resized volume at train time than at test time, teaching the model to
recognize tumors at a scale it never sees during inference. To keep
train/test preprocessing consistent, training uses the same
drop-invalid-range + resize pipeline as testing.

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
        [depth, height, width] = data.shape
        scale = [self.input_D * 1.0 / depth, self.input_H * 1.0 / height, self.input_W * 1.0 / width]
        data = ndimage.zoom(data, scale, order=0)

        return data

    def __training_data_process__(self, data, label):
        # crop data according net input size
        data = np.asanyarray(data.dataobj)
        label = np.asanyarray(label.dataobj)

        # remap STS mask foreground (255) to class label 1
        label = self.__binarize_label__(label)

        # drop out the invalid (empty background/air) range only -- no
        # tumor-centric crop, so train-time scale matches test-time scale
        data, label = self.__drop_invalid_range__(data, label)

        # random augmentation (train only): flips + mild intensity jitter.
        # Neither changes tumor scale, so train/test scale consistency is
        # preserved while still giving the model varied training samples.
        data, label = self.__random_flip__(data, label)
        data = self.__random_intensity_jitter__(data)

        # resize data
        data = self.__resize_data__(data)
        label = self.__resize_data__(label)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data, label

    def __testing_data_process__(self, data):
        # crop data according net input size
        data = np.asanyarray(data.dataobj)

        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data
