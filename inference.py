import torchvision.transforms.functional as F
import torch.nn.functional as Func
import torchvision.transforms as T
import math
import sys
import random
import time
import datetime
from typing import Iterable
import numpy as np
import PIL 
from PIL import Image
from skimage import transform
import nibabel as nib
import torch
import os
from medpy.metric.binary import dc
import pandas as pd
import glob
import re
import shutil
import copy
from skimage import measure
import math
import sys
import random
import time
import datetime
from typing import Iterable
import torch.nn.functional as Func
import numpy as np
import torch
import torch.nn as nn
import util.misc as utils
from torch.autograd import Variable
from mixup import mixup_process, get_lambda
from torch.nn import functional as F
import torchvision
import matplotlib.pyplot as plt

import util.misc as utils


def makefolder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False

def load_nii(img_path):
    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header

def save_nii(img_path, data, affine, header):
    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)

def convert_targets(targets, device):
    masks = [t["masks"] for t in targets]
    target_masks = torch.stack(masks)
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 4, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks

def conv_int(i):
    return int(i) if i.isdigit() else i

def natural_order(sord):
    if isinstance(sord, tuple):
        sord = sord[0]
    return [conv_int(c) for c in re.split(r'(\d+)', sord)]


def keep_largest_connected_components(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''
    # keep a heart connectivity 
    mask_shape = mask.shape
    
    heart_slice = np.where((mask>0), 1, 0)
    out_heart = np.zeros(heart_slice.shape, dtype=np.uint8)
    for struc_id in [1]:
        binary_img = heart_slice == struc_id
        blobs = measure.label(binary_img, connectivity=1)
        props = measure.regionprops(blobs)
        if not props:
            continue
        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label
        out_heart[blobs == largest_blob_label] = struc_id
    
    #keep LV/RV/MYO connectivity
    out_img = np.zeros(mask.shape, dtype=np.uint8)
    for struc_id in [1, 2, 3]:
        binary_img = mask == struc_id
        blobs = measure.label(binary_img, connectivity=1)
        props = measure.regionprops(blobs)
        if not props:
            continue
        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label
        out_img[blobs == largest_blob_label] = struc_id

    final_img = out_heart * out_img
    return final_img

@torch.no_grad()
def infer(model, criterion, dataloader_dict, device):
    model.eval()
    criterion.eval()
    
    test_folder = "/data/zhangke/datasets/scribble_MSCMR/val/images/"
    label_folder = "/data/zhangke/datasets/scribble_MSCMR/val/labels/"
    output_folder = "/data/zhangke/datasets/scribble_MSCMR/self_MSCMR/"

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    makefolder(output_folder)

    target_resolution = (1.36719, 1.36719)

    test_files = sorted(os.listdir(test_folder))
    label_files = sorted(os.listdir(label_folder))
    assert len(test_files) == len(label_files)

    # read_image
    for file_index in range(len(test_files)):
        test_file = test_files[file_index] 

        label_file = label_files[file_index]
        file_mask = os.path.join(label_folder, label_file)
        mask_dat = load_nii(file_mask)
        mask = mask_dat[0]

        img_path = os.path.join(test_folder, test_file)
        img_dat = load_nii(img_path)
        img = img_dat[0].copy()

        pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
        scale_vector = (pixel_size[0] / target_resolution[0],
                        pixel_size[1] / target_resolution[1])

        
        img = img.astype(np.float32)
        img = np.divide((img - np.mean(img)), np.std(img))

        slice_rescaleds = []
        for slice_index in range(img.shape[2]):
            img_slice = np.squeeze(img[:,:,slice_index])
            slice_rescaled = transform.rescale(img_slice,
                                            scale_vector,
                                            order=1,
                                            preserve_range=True,
                                            multichannel=False,
                                            anti_aliasing=True,
                                            mode='constant')
            slice_rescaleds.append(slice_rescaled)
        img = np.stack(slice_rescaleds, axis=2)

        predictions = []

        for slice_index in range(img.shape[2]):
            img_slice = img[:,:,slice_index]
            nx = 212
            ny = 212
            x, y = img_slice.shape
            x_s = (x - nx) // 2
            y_s = (y - ny) // 2
            x_c = (nx - x) // 2
            y_c = (ny - y) // 2
            # Crop section of image for prediction
            if x > nx and y > ny:
                slice_cropped = img_slice[x_s:x_s+nx, y_s:y_s+ny]
            else:
                slice_cropped = np.zeros((nx,ny))
                if x <= nx and y > ny:
                    slice_cropped[x_c:x_c+ x, :] = img_slice[:,y_s:y_s + ny]
                elif x > nx and y <= ny:
                    slice_cropped[:, y_c:y_c + y] = img_slice[x_s:x_s + nx, :]
                else:
                    slice_cropped[x_c:x_c+x, y_c:y_c + y] = img_slice[:, :]
            
            img_slice = slice_cropped
            img_slice = np.divide((slice_cropped - np.mean(slice_cropped)), np.std(slice_cropped))
            img_slice = np.reshape(img_slice, (1,1,nx,ny))

            img_slice = torch.from_numpy(img_slice)
            img_slice = img_slice.to(device)
            img_slice = img_slice.float()
            
            tasks = dataloader_dict.keys()
            task = random.sample(tasks, 1)[0]

            outputs = model(img_slice, task)
            
            softmax_out = outputs["pred_masks"]
            softmax_out = softmax_out.detach().cpu().numpy()
            prediction_cropped = np.squeeze(softmax_out[0,...])

            slice_predictions = np.zeros((4,x,y))
            # insert cropped region into original image again
            if x > nx and y > ny:
                slice_predictions[:,x_s:x_s+nx, y_s:y_s+ny] = prediction_cropped
            else:
                if x <= nx and y > ny:
                    slice_predictions[:,:, y_s:y_s+ny] = prediction_cropped[:,x_c:x_c+ x, :]
                elif x > nx and y <= ny:
                    slice_predictions[:,x_s:x_s + nx, :] = prediction_cropped[:,:, y_c:y_c + y]
                else:
                    slice_predictions[:,:, :] = prediction_cropped[:,x_c:x_c+ x, y_c:y_c + y]
            prediction = transform.resize(slice_predictions,
                                (4, mask.shape[0], mask.shape[1]),
                                order=1,
                                preserve_range=True,
                                anti_aliasing=True,
                                mode='constant')
            prediction = np.uint8(np.argmax(prediction, axis=0))
            # prediction = keep_largest_connected_components(prediction)
            predictions.append(prediction)
        prediction_arr = np.transpose(np.asarray(predictions, dtype=np.uint8), (1,2,0))
        dir_pred = os.path.join(output_folder, "predictions")
        makefolder(dir_pred)
        out_file_name = os.path.join(dir_pred, label_file)
        out_affine = mask_dat[1]
        out_header = mask_dat[2]

        save_nii(out_file_name, prediction_arr, out_affine, out_header)

        dir_gt = os.path.join(output_folder, "masks")
        makefolder(dir_gt)
        mask_file_name = os.path.join(dir_gt, label_file)
        save_nii(mask_file_name, mask_dat[0], out_affine, out_header)
    
    filenames_gt = sorted(glob.glob(os.path.join(dir_gt, '*')), key=natural_order)
    # filenames_gt = [f for f in filenames_gt if "01" in f]
    filenames_pred = sorted(glob.glob(os.path.join(dir_pred, '*')), key=natural_order)
    # filenames_pred = [f for f in filenames_pred if "01" in f]
    file_names = []
    structure_names = []
    # measures per structure:
    dices_list = []
    structures_dict = {1: 'RV', 2: 'Myo', 3: 'LV'}
    count = 0
    for p_gt, p_pred in zip(filenames_gt, filenames_pred):
        if os.path.basename(p_gt) != os.path.basename(p_pred):
            raise ValueError("The two files don't have the same name"
                            " {}, {}.".format(os.path.basename(p_gt),
                                            os.path.basename(p_pred)))
        # load ground truth and prediction
        gt, _, header = load_nii(p_gt)
        pred, _, _ = load_nii(p_pred)
        zooms = header.get_zooms()
        # calculate measures for each structure
        for struc in [3,1,2]:
            gt_binary = (gt == struc) * 1
            pred_binary = (pred == struc) * 1
            if np.sum(gt_binary) == 0 and np.sum(pred_binary) == 0:
                dices_list.append(1)
            elif np.sum(pred_binary) > 0 and np.sum(gt_binary) == 0 or np.sum(pred_binary) == 0 and np.sum(gt_binary) > 0:
                dices_list.append(0)
                count += 1
            else:
                dices_list.append(dc(gt_binary, pred_binary))
            file_names.append(os.path.basename(p_pred))
            structure_names.append(structures_dict[struc])

    df = pd.DataFrame({'dice': dices_list, 'struc': structure_names, 'filename': file_names})
    csv_path = os.path.join(output_folder, "stats.csv")
    df.to_csv(csv_path)
    return df

@torch.no_grad()
def evaluate(model, criterion, postprocessors, dataloader_dict, device, output_dir, visualizer, epoch, writer):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    #metric_logger.add_meter('loss_multiDice', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    print_freq = 10
    numbers = { k : len(v) for k, v in dataloader_dict.items() }
    iterats = { k : iter(v) for k, v in dataloader_dict.items() }
    tasks = dataloader_dict.keys()
    counts = { k : 0 for k in tasks }
    total_steps = sum(numbers.values())
    start_time = time.time()
    sample_list, output_list, target_list = [], [], []
    for step in range(total_steps):
        start = time.time()
        tasks = [ t for t in tasks if counts[t] < numbers[t] ] 
        task = random.sample(tasks, 1)[0]
        samples, targets = next(iterats[task])
        counts.update({task : counts[task] + 1 })
        datatime = time.time() - start
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in targets]

        
        targets_onehot= convert_targets(targets,device)

        #puzzlemix
        samples_var = Variable(samples.tensors, requires_grad=True)
        ###
        outputs = model(samples_var, task)
        loss_dict = criterion(outputs, targets_onehot)

        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=loss_dict_reduced_scaled['loss_CrossEntropy'], **loss_dict_reduced_scaled)
        #metric_logger.update(loss_multiDice=loss_dict_reduced['loss_multiDice'])
        itertime = time.time() - start
        metric_logger.log_every(step, total_steps, datatime, itertime, print_freq, header)
        if step % round(total_steps / 16.) == 0:  
            ##original  
            # sample_list.append(samples.tensors[0])
            ##
            ##mixup
            sample_list.append(samples_var[0])
            ##

            _, pre_masks = torch.max(outputs['pred_masks'][0], 0, keepdims=True)
            output_list.append(pre_masks)
            
            ##mixup
            target_list.append(targets_onehot.argmax(1,keepdim=True)[0])
            ##
            ##original
            # target_list.append(targets[0]['masks'])
            ##
    # gather the stats from all processes
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / total_steps))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    writer.add_scalar('avg_DSC', stats['Avg'], epoch)
    writer.add_scalar('loss_CrossEntropy', stats['loss_CrossEntropy'], epoch)
    visualizer(torch.stack(sample_list), torch.stack(output_list), torch.stack(target_list), epoch, writer)
    
    return stats