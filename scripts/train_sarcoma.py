'''
Training entry point for MNet on the Soft-tissue-Sarcoma (STS) NIfTI dataset.

This reuses MNet's model/optimizer/training loop but swaps in
SarcomaDataset (which remaps 255-valued masks to class label 1) instead of
BrainS18Dataset. The original MNet/train.py is left untouched.

NOTE on loss: sarcoma tumors occupy a tiny fraction of total image volume,
so a plain unweighted CrossEntropyLoss can reach a very low value simply by
predicting "background" everywhere. To counter this class imbalance without
overcorrecting (an earlier, too-aggressive weight of 20x pushed the model
to over-predict foreground, which also hurts Dice via poor precision), the
loss here combines:
  1. CrossEntropyLoss with a moderate weight on the foreground (tumor) class
  2. Soft Dice loss, which directly optimizes voxel overlap and is
     naturally robust to class imbalance (same idea used in nnU-Net, etc.)

NOTE on model selection: with ~40 training cases, the best-generalizing
epoch is very unlikely to be the last one -- late epochs are where
overfitting creeps in (we previously saw train Dice ~0.99 with val Dice
~0.01 using the final-epoch checkpoint). If --val_img_list is provided,
this script evaluates Dice on that list every --val_interval epochs (and
always on the final epoch) and additionally saves the best-scoring
checkpoint as <save_folder>_best.pth.tar, so you don't have to guess which
epoch to evaluate after training finishes.

Run from the MNet directory so relative imports resolve, e.g.:

    cd MNet
    python ../scripts/train_sarcoma.py --gpu_id 0 --model_depth 18 \
        --resnet_shortcut A --input_D 32 --input_H 160 --input_W 160 \
        --batch_size 1 --n_epochs 50 --dropout_prob 0.2 --pretrain_path="" \
        --data_root "C:\\path\\to\\nifti" \
        --img_list "C:\\path\\to\\nifti\\train.txt" \
        --val_img_list "C:\\path\\to\\nifti\\val.txt"
'''

import os
import sys
import time

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy import ndimage

# allow running this script from either the MNet directory or scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MNET_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "MNet")
if MNET_DIR not in sys.path:
    sys.path.insert(0, MNET_DIR)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from setting import parse_opts
from model import generate_model
from datasets.sarcoma import SarcomaDataset
from utils.logger import log
from utils.file_process import load_lines

# weight given to the tumor (foreground) class relative to background in
# CrossEntropyLoss. 20.0 was tried previously and over-corrected (model
# over-predicted foreground, hurting Dice via poor precision). A more
# moderate weight is used here, relying on the Dice loss term to do most
# of the imbalance-robustness work.
FOREGROUND_CE_WEIGHT = 6.0


def dice_loss(logits, target, smooth=1e-5):
    """
    Soft Dice loss for the foreground (tumor) class. Operates directly on
    predicted probabilities rather than hard argmax predictions, so it's
    differentiable. Robust to class imbalance since it optimizes overlap
    rather than per-voxel accuracy.
    """
    probs = F.softmax(logits, dim=1)[:, 1]  # foreground probability
    target_fg = (target == 1).float()
    intersection = (probs * target_fg).sum()
    denom = probs.sum() + target_fg.sum()
    return 1.0 - (2.0 * intersection + smooth) / (denom + smooth)


def seg_eval_dice(pred, label, smooth=1e-5):
    """Hard Dice for the foreground class, used for in-training validation."""
    pred_fg = (pred == 1)
    label_fg = (label == 1)
    inter = np.logical_and(pred_fg, label_fg).sum()
    denom = pred_fg.sum() + label_fg.sum()
    if denom == 0:
        return 1.0 if inter == 0 else 0.0
    return (2.0 * inter + smooth) / (denom + smooth)


def run_validation(model, sets, val_img_list):
    """
    Runs inference over the validation list and returns mean foreground
    Dice. Uses the same preprocessing path as scripts/test_sarcoma.py
    (SarcomaDataset in 'test' phase: trim invalid range, resize, normalize
    -- no augmentation), so numbers are directly comparable to a standalone
    test_sarcoma.py run on the same checkpoint.
    """
    import nibabel as nib

    val_sets = sets
    orig_phase = sets.phase
    val_sets.phase = 'test'

    testing_data = SarcomaDataset(sets.data_root, val_img_list, val_sets)
    data_loader = DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    img_names = [info.split(" ")[0] for info in load_lines(val_img_list)]
    label_names = [info.split(" ")[1] for info in load_lines(val_img_list)]

    model.eval()
    dices = []
    with torch.no_grad():
        for batch_id, volume in enumerate(data_loader):
            if not sets.no_cuda:
                volume = volume.cuda()
            probs = model(volume)
            probs = F.softmax(probs, dim=1)

            [_, _, mask_d, mask_h, mask_w] = probs.shape
            img = nib.load(os.path.join(sets.data_root, img_names[batch_id]))
            img = np.asanyarray(img.dataobj)
            [depth, height, width] = img.shape
            mask = probs[0].detach().cpu().numpy()
            scale = [1, depth * 1.0 / mask_d, height * 1.0 / mask_h, width * 1.0 / mask_w]
            mask = ndimage.zoom(mask, scale, order=1)
            mask = np.argmax(mask, axis=0)

            label = nib.load(os.path.join(sets.data_root, label_names[batch_id]))
            label = np.asanyarray(label.dataobj)
            label = np.where(label > 0, 1, 0)

            dices.append(seg_eval_dice(mask, label))

    model.train()
    val_sets.phase = orig_phase
    return float(np.mean(dices)) if dices else 0.0


def train(data_loader, model, optimizer, scheduler, total_epochs, save_interval, save_folder, sets):
    batches_per_epoch = len(data_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    class_weights = torch.tensor([1.0, FOREGROUND_CE_WEIGHT])
    loss_seg = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

    print("Current setting is:")
    print(sets)
    print("\n\n")
    if not sets.no_cuda:
        loss_seg = loss_seg.cuda()

    best_val_dice = -1.0
    best_epoch = -1

    model.train()
    train_time_sp = time.time()
    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))
        log.info('lr = {}'.format(scheduler.get_last_lr()))

        for batch_id, batch_data in enumerate(data_loader):
            batch_id_sp = epoch * batches_per_epoch
            volumes, label_masks = batch_data

            if not sets.no_cuda:
                volumes = volumes.cuda()

            optimizer.zero_grad()
            out_masks = model(volumes)
            [n, _, d, h, w] = out_masks.shape
            new_label_masks = np.zeros([n, d, h, w])
            for label_id in range(n):
                label_mask = label_masks[label_id]
                [ori_c, ori_d, ori_h, ori_w] = label_mask.shape
                label_mask = np.reshape(label_mask, [ori_d, ori_h, ori_w])
                scale = [d * 1.0 / ori_d, h * 1.0 / ori_h, w * 1.0 / ori_w]
                label_mask = ndimage.zoom(label_mask, scale, order=0)
                new_label_masks[label_id] = label_mask

            new_label_masks = torch.tensor(new_label_masks).to(torch.int64)
            if not sets.no_cuda:
                new_label_masks = new_label_masks.cuda()

            loss_value_seg = loss_seg(out_masks, new_label_masks)
            loss_value_dice = dice_loss(out_masks, new_label_masks)
            loss = loss_value_seg + loss_value_dice
            loss.backward()
            optimizer.step()

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                'Batch: {}-{} ({}), loss = {:.3f}, loss_seg = {:.3f}, loss_dice = {:.3f}, avg_batch_time = {:.3f}'
                .format(epoch, batch_id, batch_id_sp, loss.item(), loss_value_seg.item(),
                        loss_value_dice.item(), avg_batch_time))

            if not sets.ci_test:
                if batch_id == 0 and batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                    model_save_path = '{}_epoch_{}_batch_{}.pth.tar'.format(save_folder, epoch, batch_id)
                    model_save_dir = os.path.dirname(model_save_path)
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)

                    log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id))
                    torch.save({
                        'epoch': epoch,
                        'batch_id': batch_id,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                        model_save_path)

        # scheduler steps once per epoch (not once per batch) so a
        # 0.99 decay factor corresponds to ~1% decay per epoch, not per
        # batch. Stepping per-batch was causing lr to collapse to ~0
        # after only a handful of epochs, effectively freezing training.
        scheduler.step()

        # always save a checkpoint at the end of each epoch so a full run
        # is never lost if save_interval doesn't line up with batches_per_epoch
        model_save_path = '{}_epoch_{}_final.pth.tar'.format(save_folder, epoch)
        model_save_dir = os.path.dirname(model_save_path)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()},
            model_save_path)

        # optional in-training validation: evaluate Dice periodically (and
        # always on the last epoch) and track/save the best checkpoint, so
        # we don't have to guess which epoch generalizes best after the
        # fact.
        is_last_epoch = (epoch == total_epochs - 1)
        if sets.val_img_list and ((epoch + 1) % sets.val_interval == 0 or is_last_epoch):
            val_dice = run_validation(model, sets, sets.val_img_list)
            log.info('Epoch {}: validation mean Dice (class 1) = {:.4f}'.format(epoch, val_dice))
            print('Epoch {}: validation mean Dice (class 1) = {:.4f}'.format(epoch, val_dice))
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_epoch = epoch
                best_save_path = '{}_best.pth.tar'.format(save_folder)
                torch.save({
                    'epoch': epoch,
                    'val_dice': val_dice,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    best_save_path)
                log.info('New best checkpoint: epoch {}, val Dice {:.4f} -> {}'
                          .format(epoch, val_dice, best_save_path))

    if sets.val_img_list:
        print('Best validation Dice: {:.4f} at epoch {} (saved to {}_best.pth.tar)'
              .format(best_val_dice, best_epoch, save_folder))

    print('Finished training')
    if sets.ci_test:
        exit()


if __name__ == '__main__':
    sets = parse_opts()
    sets.save_folder = "./trails/models/sarcoma_{}_{}".format(sets.model, sets.model_depth)

    torch.manual_seed(sets.manual_seed)
    model, parameters = generate_model(sets)
    print(model)

    if isinstance(parameters, dict):
        params = [
            {'params': parameters['base_parameters'], 'lr': sets.learning_rate},
            {'params': parameters['new_parameters'], 'lr': sets.learning_rate * 100}
        ]
    else:
        params = [{'params': parameters, 'lr': sets.learning_rate}]
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    if sets.resume_path:
        if os.path.isfile(sets.resume_path):
            print("=> loading checkpoint '{}'".format(sets.resume_path))
            checkpoint = torch.load(sets.resume_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(sets.resume_path, checkpoint['epoch']))

    sets.phase = 'train'
    if sets.no_cuda:
        sets.pin_memory = False
    else:
        sets.pin_memory = True
    training_dataset = SarcomaDataset(sets.data_root, sets.img_list, sets)
    data_loader = DataLoader(training_dataset, batch_size=sets.batch_size, shuffle=True,
                              num_workers=sets.num_workers, pin_memory=sets.pin_memory)

    train(data_loader, model, optimizer, scheduler, total_epochs=sets.n_epochs,
          save_interval=sets.save_intervals, save_folder=sets.save_folder, sets=sets)
