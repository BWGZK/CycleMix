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
from cutout import Cutout, rotate_invariant, rotate_back
from inference import keep_largest_connected_components

class Visualize_train(nn.Module):
    def __init__(self):
        super().__init__()
        
    def save_image(self, image, tag, epoch, writer):
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)
        
    def forward(self, originals, inputs, outputs, ori_labels, labels, mixed_labels, epoch, writer):
        self.save_image(originals, 'inputs_original', epoch, writer)
        self.save_image(inputs, 'inputs_train', epoch, writer)
        self.save_image(outputs.float(), 'outputs_train', epoch, writer)
        self.save_image(labels.float(), 'labels_train', epoch, writer)
        self.save_image(mixed_labels.float(), 'labels_mixed', epoch, writer)
        self.save_image(ori_labels.float(), 'labels_original', epoch, writer)


def convert_targets(targets, device):
    masks = [t["masks"] for t in targets]
    target_masks = torch.stack(masks)
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 5, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks

def to_onehot(target_masks, device):
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 5, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks


def to_onehot_dim4(target_masks, device):
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 4, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    dataloader_dict: dict, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, args, writer):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    numbers = { k : len(v) for k, v in dataloader_dict.items() }
    iterats = { k : iter(v) for k, v in dataloader_dict.items() }
    tasks = dataloader_dict.keys()
    counts = { k : 0 for k in tasks }
    total_steps = sum(numbers.values())
    start_time = time.time()
    original_list,sample_list, output_list, target_list, target_ori_list, target_mixed_list =[], [], [], [], [], []
    for step in range(total_steps):
        start = time.time()
        tasks = [ t for t in tasks if counts[t] < numbers[t] ]
        task = random.sample(tasks, 1)[0]
        samples, targets = next(iterats[task])

        counts.update({task : counts[task] + 1 })
        datatime = time.time() - start
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in targets]

        targets_onehot = convert_targets(targets,device)

        # puzzlemix
        samples_var = Variable(samples.tensors, requires_grad=True)
        # puzzlemix -- parameters
        adv_p = 0.1 
        adv_eps = 10.0

        adv_mask1 = np.random.binomial(n=1, p=adv_p)
        adv_mask2 = np.random.binomial(n=1, p=adv_p)

        noise=None
        if (adv_mask1 == 1 or adv_mask2 == 1):
            noise = torch.zeros_like(samples_var).uniform_(adv_eps/255., adv_eps/255.)             
            input_noise = samples_var + noise
            samples_var = Variable(input_noise, requires_grad=True)
 
        ###

        # puzzlemix -- backward
        outputs = model(samples_var, task)
        loss_dict = criterion(outputs, targets_onehot)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in ['loss_CrossEntropy'] if k in weight_dict)
        losses.backward(retain_graph=True)
        ###

        ### original output with unmixed input image:
        output_original = model(samples_var,task)
        ###

        # puzzlemix -- calculate unary
        unary = torch.sqrt(torch.mean(samples_var.grad **2, dim=1))
        unary = F.pad(unary, (22,22,22,22,0,0), 'constant')
        ###

        # puzzlemix -- calculate adversarial noise
        if (adv_mask1 == 1 or adv_mask2 == 1):
            noise += (adv_eps + 2) / 255. * samples_var.grad.sign()
            noise = torch.clamp(noise, -args.adv_eps/255., args.adv_eps/255.)
            adv_mix_coef = np.random.uniform(0,1)
            noise = adv_mix_coef * noise

        samples_var_256 = F.pad(samples_var, (22,22,22,22,0,0,0,0), 'constant')
        targets_onehot_256 = F.pad(targets_onehot, (22,22,22,22,0,0,0,0), 'constant')
        out, reweighted_target, indices_transport, mask_transport = mixup_process(samples_var_256, targets_onehot_256, args=args, grad= unary, noise = noise)
        out = out[:,:,22:-22,22:-22]
        reweighted_target = reweighted_target[:,:,22:-22,22:-22]
        mask_transport = mask_transport[:,:,22:-22,22:-22]
        ###

        #Cutout
        samples_cut, targets_cut, masks_cut = Cutout(out, reweighted_target, device)
        ###

        #rotate back
        samples_cut, targets_cut, angles = rotate_invariant(samples_cut, targets_cut)
        masks_cut = masks_cut.to(device)
        outputs_cut = model(samples_cut, task)
        samples_cut_back, outputs_cut,targets_cut = rotate_back(samples_cut, outputs_cut["pred_masks"],targets_cut,angles)
        ###

        # cutout_loss
        loss_dict_cut = criterion(outputs_cut, targets_cut)
        losses_cut = sum(loss_dict_cut[k] * weight_dict[k] for k in loss_dict_cut.keys() if k in ['loss_CrossEntropy'])
        if step == 0:
            print("cutout loss:", losses_cut.item())
        ###

        ### mixed output
        mixed_output = torch.zeros_like(outputs_cut["pred_masks"])
        shuffled_output = output_original["pred_masks"][indices_transport].clone()
        for i in range(shuffled_output.shape[1]):
            mixed_output[:,i,:,:] = output_original["pred_masks"][:,i,:,:] * mask_transport[:,0,:,:] + shuffled_output[:,i,:,:] * (1-mask_transport[:,0,:,:])
        mixed_output = mixed_output*masks_cut
        ###

        # save visualize images
        if step % 200 == 0:
            for i in range(samples_var.shape[0]):  
                original_list.append(samples_var[i])
                sample_list.append(samples_cut_back[i])
                _, pre_masks = torch.max(outputs_cut['pred_masks'][i], 0, keepdims=True)
                output_list.append(pre_masks)
                target_ori_list.append(targets_onehot.argmax(1,keepdim=True)[i])
                target_list.append(targets_cut.argmax(1,keepdim=True)[i])
                target_mixed_list.append(mixed_output.argmax(1,keepdim=True)[i])
        ###

        # supervised loss for unmixed images
        loss_dict_ori = criterion(output_original, targets_onehot)
        weight_dict = criterion.weight_dict
        losses_ori = sum(loss_dict_ori[k] * weight_dict[k] for k in ['loss_CrossEntropy'] if k in weight_dict)
        if step == 0:
            print("original loss:", losses_ori.item())
        ###

        # integrity loss for unmixed images
        original_masks = output_original["pred_masks"]
        predictions_original_list = []
        
        for i in range(original_masks.shape[0]):
            prediction = np.uint8(np.argmax(original_masks[i,:,:,:].detach().cpu(), axis=0))
            prediction = keep_largest_connected_components(prediction)
            prediction = torch.from_numpy(prediction).to(device)
            predictions_original_list.append(prediction)

        predictions = torch.stack(predictions_original_list)
        predictions = torch.unsqueeze(predictions, 1)
        prediction_onehot = to_onehot_dim4(predictions,device)
        losses_integrity = 1- Func.cosine_similarity(original_masks[:,0:4,:,:], prediction_onehot, dim=1).mean()
        
        losses_integrity = 0.1*losses_integrity
        if step == 0:
            print("integrity loss:", losses_integrity.item())
        ###

        # invariant_loss
        invariant_loss = 1- Func.cosine_similarity(outputs_cut["pred_masks"], mixed_output, dim=1).mean()
        invariant_loss = 0.1*invariant_loss
        if step == 0:
            print("invariant loss:", invariant_loss.item())
        ###

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in ['loss_CrossEntropy']}
        optimizer.zero_grad()

        losses_final = losses_ori+losses_integrity+invariant_loss+losses_cut
        losses_final.backward()

        optimizer.step()
        metric_logger.update(loss=loss_dict_reduced_scaled['loss_CrossEntropy'], **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        itertime = time.time() - start
        metric_logger.log_every(step, total_steps, datatime, itertime, print_freq, header)

    # gather the stats from all processes
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / total_steps))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    visual_train = Visualize_train()
    visual_train(torch.stack(original_list),torch.stack(sample_list), torch.stack(output_list), torch.stack(target_ori_list),torch.stack(target_list), torch.stack(target_mixed_list), epoch, writer)
    
    return stats