'''
Testing/evaluation entry point for MNet on the Soft-tissue-Sarcoma (STS)
NIfTI dataset. Computes mean Dice score for the foreground (tumor) class.

Uses SarcomaDataset for consistent mask handling (255 -> 1 remap) during
ground-truth loading.

Run from the MNet directory so relative imports resolve, e.g.:

    cd MNet
    python ../scripts/test_sarcoma.py --gpu_id 0 --model_depth 18 \
        --resnet_shortcut A --input_D 32 --input_H 160 --input_W 160 \
        --resume_path trails/models/sarcoma_resnet_18_epoch_29_final.pth.tar \
        --data_root "C:\\path\\to\\nifti" \
        --img_list "C:\\path\\to\\nifti\\val.txt"
'''

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy import ndimage
import nibabel as nib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MNET_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "MNet")
if MNET_DIR not in sys.path:
    sys.path.insert(0, MNET_DIR)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from setting import parse_opts
from model import generate_model
from datasets.sarcoma import SarcomaDataset
from utils.file_process import load_lines


def seg_eval(pred, label, clss):
    """
    calculate the dice between prediction and ground truth
    input:
        pred: predicted mask
        label: groud truth
        clss: eg. [0, 1] for binary class
    """
    Ncls = len(clss)
    dices = np.zeros(Ncls)
    [depth, height, width] = pred.shape
    for idx, cls in enumerate(clss):
        pred_cls = np.zeros([depth, height, width])
        pred_cls[np.where(pred == cls)] = 1
        label_cls = np.zeros([depth, height, width])
        label_cls[np.where(label == cls)] = 1

        s = pred_cls + label_cls
        inter = len(np.where(s >= 2)[0])
        conv = len(np.where(s >= 1)[0]) + inter
        try:
            dice = 2.0 * inter / conv
        except Exception:
            print("conv is zeros when dice = 2.0 * inter / conv")
            dice = -1

        dices[idx] = dice

    return dices


def test(data_loader, model, img_names, sets):
    masks = []
    model.eval()
    for batch_id, batch_data in enumerate(data_loader):
        volume = batch_data
        if not sets.no_cuda:
            volume = volume.cuda()
        with torch.no_grad():
            probs = model(volume)
            probs = F.softmax(probs, dim=1)

        [batchsize, _, mask_d, mask_h, mask_w] = probs.shape
        data = nib.load(os.path.join(sets.data_root, img_names[batch_id]))
        data = np.asanyarray(data.dataobj)
        [depth, height, width] = data.shape
        mask = probs[0].detach().cpu().numpy()
        scale = [1, depth * 1.0 / mask_d, height * 1.0 / mask_h, width * 1.0 / mask_w]
        mask = ndimage.zoom(mask, scale, order=1)
        mask = np.argmax(mask, axis=0)

        masks.append(mask)

    return masks


if __name__ == '__main__':
    sets = parse_opts()
    sets.target_type = "normal"
    sets.phase = 'test'

    checkpoint = torch.load(sets.resume_path)
    net, _ = generate_model(sets)
    net.load_state_dict(checkpoint['state_dict'])

    testing_data = SarcomaDataset(sets.data_root, sets.img_list, sets)
    data_loader = DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

    img_names = [info.split(" ")[0] for info in load_lines(sets.img_list)]
    masks = test(data_loader, net, img_names, sets)

    label_names = [info.split(" ")[1] for info in load_lines(sets.img_list)]
    Nimg = len(label_names)
    dices = np.zeros([Nimg, sets.n_seg_classes])
    for idx in range(Nimg):
        label = nib.load(os.path.join(sets.data_root, label_names[idx]))
        label = np.asanyarray(label.dataobj)
        # remap STS mask foreground (255) to class label 1, matching training
        label = np.where(label > 0, 1, 0)
        dices[idx, :] = seg_eval(masks[idx], label, range(sets.n_seg_classes))

    for idx in range(1, sets.n_seg_classes):
        mean_dice_per_task = np.mean(dices[:, idx])
        print('mean dice for class-{} is {}'.format(idx, mean_dice_per_task))
