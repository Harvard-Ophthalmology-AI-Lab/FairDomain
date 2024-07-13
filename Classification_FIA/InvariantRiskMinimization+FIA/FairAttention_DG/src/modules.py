import sys, os

import blobfile as bf
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from torchvision.models import *
import torch.nn.functional as F

from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ExponentialLR, StepLR

from sklearn.utils import *
from sklearn.metrics import *
from fairlearn.metrics import *

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if len(output.shape) == 1:
        acc = np.sum((output >= 0.5).astype(float) == target)/target.shape[0]
        return acc.item()

    if isinstance(output, np.ndarray):
        output = torch.tensor(output)
    if isinstance(target, np.ndarray):
        target = torch.tensor(target)
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, dim=1)
        target = target.view(batch_size, 1).repeat(1, maxk)
        
        correct = (pred == target)
  
        topk_accuracy = []
        for k in topk:
            accuracy = correct[:, :k].float().sum().item()
            accuracy /= batch_size # [0, 1.]
            topk_accuracy.append(accuracy)
        
        return topk_accuracy[0]

def compute_auc(pred_prob, y, num_classes=2):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    if num_classes == 2:
        fpr, tpr, thresholds = roc_curve(y, pred_prob)
        auc_val = auc(fpr, tpr)
    elif num_classes > 2:
        y_onehot = num_to_onehot(y, num_classes)
        auc_val = roc_auc_score(y_onehot, pred_prob, average='macro', multi_class='ovr')

    return auc_val

def auc_score(pred_prob, y):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    if np.unique(y).shape[0]>2:
        AUC = roc_auc_score(y, pred_prob, multi_class='ovr')
    else:
        fpr, tpr, thresholds = roc_curve(y, pred_prob)
        AUC = auc(fpr, tpr)
    
    return AUC

def num_to_onehot(nums, num_to_class):
    nums = nums.astype(int)
    n_values = num_to_class
    onehot_vec = np.eye(n_values)[nums]
    return onehot_vec

def prob_to_label(pred_prob):
    # Find the indices of the highest probabilities for each sample
    max_prob_indices = np.argmax(pred_prob, axis=1)

    # Create one-hot vectors for each sample
    one_hot_vectors = np.zeros_like(pred_prob)
    one_hot_vectors[np.arange(len(max_prob_indices)), max_prob_indices] = 1

    return one_hot_vectors

def numeric_to_one_hot(y, num_classes=None):
    y = np.asarray(y, dtype=np.int32)

    if num_classes is None:
        num_classes = np.max(y) + 1
    
    one_hot_array = np.zeros((len(y), num_classes))
    one_hot_array[np.arange(len(y)), y] = 1
    
    return one_hot_array

def multiclass_demographic_parity(pred_prob, y, attrs):

    pred_one_hot = prob_to_label(pred_prob)

    gt_one_hot = numeric_to_one_hot(y)

    scores = []
    for i in range(pred_one_hot.shape[1]):
        tmp_score = demographic_parity_difference(pred_one_hot[:,i],
                                gt_one_hot[:,i],
                                sensitive_features=attrs)

        scores.append(tmp_score)

    avg_score = np.mean(scores)
        
    return avg_score

def multiclass_equalized_odds(pred_prob, y, attrs):

    pred_one_hot = prob_to_label(pred_prob)

    gt_one_hot = numeric_to_one_hot(y)

    scores = []
    for i in range(pred_one_hot.shape[1]):
        tmp_score = equalized_odds_difference(pred_one_hot[:,i],
                            gt_one_hot[:,i],
                            sensitive_features=attrs)

        scores.append(tmp_score)

    avg_score = np.mean(scores)
        
    return avg_score

def multiclass_demographic_parity_(pred_prob, y, attrs):

    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    attrs_set = np.unique(attrs)
    y_pred = np.argmax(pred_prob, axis=1)

    mc_dpd = 0
    for i in range(pred_prob.shape[1]):
        tmp_preds = (y_pred==i).astype(int)
        tmp_not_preds = 1 - tmp_preds

        dp_by_attrs = []
        for j in attrs_set:
            idx = attrs==j
            tmp = np.abs(tmp_preds.mean().item() - tmp_preds[idx].mean().item()) + np.abs(tmp_not_preds.mean().item() - tmp_not_preds[idx].mean().item())
            dp_by_attrs.append(tmp)
            print(tmp)
        mc_dpd += np.mean(dp_by_attrs).item()

    mc_dpd = mc_dpd / pred_prob.shape[1]
        
    return mc_dpd

def auc_score_multiclass(pred_prob, y, num_of_class=3, eps=0.01):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    sensitivity_at_diff_specificity = [-1]*4
    y_onehot = num_to_onehot(y, num_of_class)
    fpr, tpr, thresholds = roc_curve(y_onehot.ravel(), pred_prob.ravel())
    for i in range(len(fpr)):
        cur_fpr = fpr[i]
        cur_tpr = tpr[i]
        if np.abs(cur_fpr-0.2) <= eps:
            sensitivity_at_diff_specificity[0] = cur_tpr
        if np.abs(cur_fpr-0.15) <= eps:
            sensitivity_at_diff_specificity[1] = cur_tpr
        if np.abs(cur_fpr-0.1) <= eps:
            sensitivity_at_diff_specificity[2] = cur_tpr
        if np.abs(cur_fpr-0.05) <= eps:
            sensitivity_at_diff_specificity[3] = cur_tpr
    AUC = auc(fpr, tpr)
    
    return AUC, sensitivity_at_diff_specificity

def equity_scaled_accuracy(output, target, attrs, alpha=1.):
    es_acc = 0
    if len(output.shape) >= 2:
        overall_acc = np.sum(np.argmax(output, axis=1) == target)/target.shape[0]
    else:
        overall_acc = np.sum((output >= 0.5).astype(float) == target)/target.shape[0]
    tmp = 0
    identity_wise_perf = []
    identity_wise_num = []
    for one_attr in np.unique(attrs).astype(int):
        pred_group = output[attrs == one_attr]
        gt_group = target[attrs == one_attr]

        if len(pred_group.shape) >= 2:
            acc = np.sum(np.argmax(pred_group, axis=1) == gt_group)/gt_group.shape[0]
        else:
            acc = np.sum((pred_group >= 0.5).astype(float) == gt_group)/gt_group.shape[0]

        identity_wise_perf.append(acc)
        identity_wise_num.append(gt_group.shape[0])

    for i in range(len(identity_wise_perf)):
        tmp += np.abs(identity_wise_perf[i]-overall_acc)
    es_acc = (overall_acc / (alpha*tmp + 1))
    
    return es_acc

def equity_scaled_AUC(output, target, attrs, alpha=1., num_classes=2):
    es_auc = 0
    tmp = 0
    identity_wise_perf = []
    identity_wise_num = []
    
    if num_classes == 2:
        fpr, tpr, thresholds = roc_curve(target, output)
        overall_auc = auc(fpr, tpr)
    elif num_classes > 2:
        y_onehot = num_to_onehot(target, num_classes)
        overall_auc = roc_auc_score(y_onehot, output, average='macro', multi_class='ovr')

    for one_attr in np.unique(attrs).astype(int):
        pred_group = output[attrs == one_attr]
        gt_group = target[attrs == one_attr]

        if num_classes == 2:
            fpr, tpr, thresholds = roc_curve(gt_group, pred_group)
            group_auc = auc(fpr, tpr)
        elif num_classes > 2:
            y_onehot = num_to_onehot(gt_group, num_classes)
            group_auc = roc_auc_score(y_onehot, pred_group, average='macro', multi_class='ovr')
        
        identity_wise_perf.append(group_auc)
        identity_wise_num.append(gt_group.shape[0])

    for i in range(len(identity_wise_perf)):
        tmp += np.abs(identity_wise_perf[i]-overall_auc)
    es_auc = (overall_auc / (alpha*tmp + 1))

    return es_auc

def evalute_perf_by_attr(preds, gts, attrs=None, num_classes=2):

    esaccs_by_attrs = []
    esaucs_by_attrs = []
    aucs_by_attrs = []
    dpds = []
    dprs = []
    eods = []
    eors = []
    for i in range(attrs.shape[0]):
        attr = attrs[i,:]

        es_acc = equity_scaled_accuracy(preds, gts, attr)
        esaccs_by_attrs.append(es_acc)
        es_auc = equity_scaled_AUC(preds, gts, attr, num_classes=num_classes)
        esaucs_by_attrs.append(es_auc)

        aucs_by_group = []
        elements = np.unique(attr).astype(int)
        for e in elements:
            aucs_by_group.append( compute_auc(preds[attr == e], gts[attr == e], num_classes=num_classes) )
        aucs_by_attrs.append(aucs_by_group)
        pred_labels = (preds >= 0.5).astype(float)
        if num_classes == 2:
            dpd = demographic_parity_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            dpr = demographic_parity_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eod = equalized_odds_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eor = equalized_odds_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
        elif num_classes > 2:
            dpd = multiclass_demographic_parity(preds, gts, attr)
            dpr = 0
            eod = multiclass_equalized_odds(preds, gts, attr)
            eor = 0

        dpds.append(dpd)
        eods.append(eod)

    return esaccs_by_attrs, esaucs_by_attrs, aucs_by_attrs, dpds, eods


def evalute_comprehensive_perf(preds, gts, attrs=None, num_classes=2):

    esaccs_by_attrs = []
    esaucs_by_attrs = []
    aucs_by_attrs = []
    dpds = []
    dprs = []
    eods = []
    eors = []
    between_group_disparity = []

    overall_auc = compute_auc(preds, gts, num_classes=num_classes)

    for i in range(attrs.shape[0]):
        attr = attrs[i,:]

        es_acc = equity_scaled_accuracy(preds, gts, attr)
        esaccs_by_attrs.append(es_acc)

        try:
            es_auc = equity_scaled_AUC(preds, gts, attr, num_classes=num_classes)
        except Exception as e:
            es_auc = -1.
        esaucs_by_attrs.append(es_auc)

        aucs_by_group = []
        elements = np.unique(attr).astype(int)
        for e in elements:
            
            if e < 0:
                continue
            
            try:
                tmp_auc = compute_auc(preds[attr == e], gts[attr == e], num_classes=num_classes)
            except Exception as e:
                tmp_auc = -1.
            aucs_by_group.append( tmp_auc )
        aucs_by_attrs.append(aucs_by_group)
        std_disparity, max_disparity = compute_between_group_disparity(aucs_by_group, overall_auc)
        between_group_disparity.append([std_disparity, max_disparity])

        pred_labels = (preds >= 0.5).astype(float)
        if num_classes == 2:
            dpd = demographic_parity_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            dpr = demographic_parity_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eod = equalized_odds_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eor = equalized_odds_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
        elif num_classes > 2:
            dpd = multiclass_demographic_parity(preds, gts, attr)
            dpr = 0
            eod = multiclass_equalized_odds(preds, gts, attr)
            eor = 0

        dpds.append(dpd)
        eods.append(eod)

    return esaccs_by_attrs, esaucs_by_attrs, aucs_by_attrs, dpds, eods, between_group_disparity

def evalute_comprehensive_perf_(preds, gts, attrs=None, num_classes=2):

    esaccs_by_attrs = []
    esaucs_by_attrs = []
    aucs_by_attrs = []
    dpds = []
    dprs = []
    eods = []
    eors = []
    between_group_disparity = []

    overall_auc = compute_auc(preds, gts, num_classes=num_classes)

    for i in range(attrs.shape[0]):
        attr = attrs[i,:]
        
        es_acc = equity_scaled_accuracy(preds, gts, attr)
        esaccs_by_attrs.append(es_acc)
        es_auc = equity_scaled_AUC(preds, gts, attr, num_classes=num_classes)
        esaucs_by_attrs.append(es_auc)

        aucs_by_group = []
        elements = np.unique(attr).astype(int)
        
        for e in elements:
            aucs_by_group.append( compute_auc(preds[attr == e], gts[attr == e], num_classes=num_classes) )
        aucs_by_attrs.append(aucs_by_group)
        std_disparity, max_disparity = compute_between_group_disparity_half(aucs_by_group, overall_auc)
        between_group_disparity.append([std_disparity, max_disparity])

        pred_labels = (preds >= 0.5).astype(float)
        if num_classes == 2:
            dpd = demographic_parity_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            dpr = demographic_parity_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eod = equalized_odds_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eor = equalized_odds_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
        elif num_classes > 2:
            dpd = multiclass_demographic_parity(preds, gts, attr)
            dpr = 0
            eod = multiclass_equalized_odds(preds, gts, attr)
            eor = 0

        dpds.append(dpd)
        eods.append(eod)

    return esaccs_by_attrs, esaucs_by_attrs, aucs_by_attrs, dpds, eods, between_group_disparity

def evalute_comprehensive_perf_scores(preds, gts, attrs=None, num_classes=2):

    esaccs_by_attrs = []
    esaucs_by_attrs = []
    aucs_by_attrs = []
    dpds = []
    dprs = []
    eods = []
    eors = []
    between_group_disparity = []

    overall_acc = accuracy(preds, gts, topk=(1,))
    overall_auc = compute_auc(preds, gts, num_classes=num_classes)

    for i in range(attrs.shape[0]):
        attr = attrs[i,:]
        
        es_acc = equity_scaled_accuracy(preds, gts, attr)
        esaccs_by_attrs.append(es_acc)
        # es_auc = equity_scaled_AUC(preds, gts, attr, num_classes=num_classes)
        try:
            es_auc = equity_scaled_AUC(preds, gts, attr, num_classes=num_classes)
        except Exception as e:
            es_auc = -1.
        esaucs_by_attrs.append(es_auc)

        aucs_by_group = []
        elements = np.unique(attr).astype(int)
        for e in elements:
            try:
                tmp_auc = compute_auc(preds[attr == e], gts[attr == e], num_classes=num_classes)
            except Exception as e:
                tmp_auc = -1.
            aucs_by_group.append( tmp_auc )
        aucs_by_attrs.append(np.array(aucs_by_group))
        std_disparity, max_disparity = compute_between_group_disparity(aucs_by_group, overall_auc)
        between_group_disparity.append([std_disparity, max_disparity])

        pred_labels = (preds >= 0.5).astype(float)
        if num_classes == 2:
            dpd = demographic_parity_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            dpr = demographic_parity_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eod = equalized_odds_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eor = equalized_odds_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
        elif num_classes > 2:
            dpd = multiclass_demographic_parity(preds, gts, attr)
            dpr = 0
            eod = multiclass_equalized_odds(preds, gts, attr)
            eor = 0

        dpds.append(dpd)
        eods.append(eod)

    esaccs_by_attrs = np.array(esaccs_by_attrs)
    esaucs_by_attrs = np.array(esaucs_by_attrs)
    # aucs_by_attrs = np.array(aucs_by_attrs, dtype=object).astype(np.float)
    dpds = np.array(dpds)
    eods = np.array(eods)
    between_group_disparity = np.array(between_group_disparity)

    return overall_acc, esaccs_by_attrs, overall_auc, esaucs_by_attrs, aucs_by_attrs, dpds, eods, between_group_disparity

def bootstrap_performance(test_preds, test_gts, test_attrs, bootstrap_repeat_times=100, num_classes=2, num_attrs=3):
    test_acc, test_es_acc, test_auc, test_es_auc, test_aucs_by_attrs, test_dpds, test_eods, test_between_group_disparity = [], [], [], [], [[] for i in range(num_attrs)], [], [], []
    for i in range(bootstrap_repeat_times):
        tmp_indices = np.array(list(range(0, test_gts.shape[0])))
        bootstrap_indices = resample(tmp_indices, replace=True, n_samples=test_gts.shape[0])
        # bootstrap_pairs_array1, bootstrap_pairs_array2 = resample(preds, gts, replace=True, n_samples=gts.shape[0]) #, n_samples=sample_size, random_state=random_state
        bootstrap_preds = test_preds[bootstrap_indices]
        bootstrap_gts = test_gts[bootstrap_indices]
        bootstrap_attrs = []
        for x in test_attrs:
            bootstrap_attrs.append(x[bootstrap_indices])
        bootstrap_attrs = np.vstack(bootstrap_attrs)
        tmp_test_acc, tmp_test_es_acc, tmp_test_auc, tmp_test_es_auc, tmp_test_aucs_by_attrs, tmp_test_dpds, tmp_test_eods, tmp_test_between_group_disparity = evalute_comprehensive_perf_scores(bootstrap_preds, bootstrap_gts, bootstrap_attrs, num_classes=num_classes)
        test_acc.append(tmp_test_acc)
        test_es_acc.append(tmp_test_es_acc)
        test_auc.append(tmp_test_auc)
        test_es_auc.append(tmp_test_es_auc)
        for j in range(len(test_aucs_by_attrs)):
            test_aucs_by_attrs[j].append(tmp_test_aucs_by_attrs[j])
        test_dpds.append(tmp_test_dpds)
        test_eods.append(tmp_test_eods)
        test_between_group_disparity.append(tmp_test_between_group_disparity)
    test_acc = np.vstack(test_acc)
    test_es_acc = np.vstack(test_es_acc)
    test_auc = np.vstack(test_auc)
    test_es_auc = np.vstack(test_es_auc)
    for j in range(len(test_aucs_by_attrs)):
        test_aucs_by_attrs[j] = np.vstack(test_aucs_by_attrs[j])
    test_dpds = np.vstack(test_dpds)
    test_eods = np.vstack(test_eods)
    test_between_group_disparity = np.array(test_between_group_disparity)[None,:,:]
    test_between_group_disparity = np.vstack(test_between_group_disparity)

    acc = np.mean(test_acc, axis=0)[0]
    es_acc = np.mean(test_es_acc, axis=0)
    auc  = np.mean(test_auc, axis=0)[0]
    es_auc = np.mean(test_es_auc, axis=0)
    # aucs_by_attrs = np.mean(test_aucs_by_attrs, axis=0)
    aucs_by_attrs = []
    for j in range(len(test_aucs_by_attrs)):
        aucs_by_attrs.append(np.mean(test_aucs_by_attrs[j], axis=0))
    dpds = np.mean(test_dpds, axis=0)
    eods = np.mean(test_eods, axis=0)
    between_group_disparity = np.mean(test_between_group_disparity, axis=0)

    acc_std = np.std(test_acc, axis=0)[0]
    es_acc_std = np.std(test_es_acc, axis=0)
    auc_std = np.std(test_auc, axis=0)[0]
    es_auc_std = np.std(test_es_auc, axis=0)
    # aucs_by_attrs_std = np.std(test_aucs_by_attrs, axis=0)
    aucs_by_attrs_std = []
    for j in range(len(test_aucs_by_attrs)):
        aucs_by_attrs_std.append(np.std(test_aucs_by_attrs[j], axis=0))
    dpds_std = np.std(test_dpds, axis=0)
    eods_std = np.std(test_eods, axis=0)
    between_group_disparity_std = np.std(test_between_group_disparity, axis=0)

    return acc, es_acc, auc, es_auc, aucs_by_attrs, dpds, eods, between_group_disparity, \
            acc_std, es_acc_std, auc_std, es_auc_std, aucs_by_attrs_std, dpds_std, eods_std, between_group_disparity_std

def scale_losses(loss_tensor, attr=None, level='individual', fair_scaling_beta=1., 
                    fair_scaling_temperature=1., fair_scaling_coefs=[0.5, 0.5], fair_scaling_minority=[1, 1, 0]):

    if level == 'individual':
        tmp_loss = loss_tensor ** fair_scaling_beta / fair_scaling_temperature
        loss = ( (torch.exp(tmp_loss) / torch.exp(tmp_loss).sum())*loss_tensor.shape[0]*loss_tensor ).mean()
    elif level == 'group':
        tmp_weights = torch.zeros(int(torch.max(attr).item())+1).type(loss_tensor.type())
        for x in attr:
            tmp_weights[x.long()] = torch.mean(loss_tensor[attr==x] ** fair_scaling_beta / (1+torch.abs(loss_tensor[attr==x])) ).item()
        tmp_weights = torch.softmax(tmp_weights/fair_scaling_temperature, dim=0)
        tmp_weights_inplace = tmp_weights[attr.long()]*loss_tensor.shape[0]
        loss = (tmp_weights_inplace * loss_tensor).mean()
    elif level == 'individual+group':
        tmp_loss = (loss_tensor+fair_scaling_coefs[fair_scaling_minority[attr[i].long()]]) ** fair_scaling_beta / fair_scaling_temperature
        tmp_weights = torch.zeros(loss_tensor.shape[0]).type(loss_tensor.type())
        for i in range(len(loss_tensor)):
            tmp_weights[i] = fair_scaling_coefs[fair_scaling_minority[attr[i].long()]] * tmp_loss[i]
        tmp_weights = torch.softmax(tmp_weights, dim=0)*loss_tensor.shape[0]
        loss = (tmp_weights * loss_tensor).mean()

    return loss

class Fair_Loss_Scaler(nn.Module):
    def __init__(self, level='individual', fair_scaling_group_weights=None, 
                    fair_scaling_temperature=1., fair_scaling_coef=.5):
        super().__init__()
        self.level = level
        self.fair_scaling_temperature = fair_scaling_temperature
        self.fair_scaling_coef = fair_scaling_coef
        self.fair_scaling_group_weights = nn.Parameter(torch.tensor(fair_scaling_group_weights))

    def forward(self, x, attr):
        
        tmp_loss = ((1-self.fair_scaling_coef)*x + self.fair_scaling_coef*self.fair_scaling_group_weights[attr.long()]) / self.fair_scaling_temperature
        tmp_weights = torch.softmax(tmp_loss, dim=0) * x.shape[0]
        loss = (tmp_weights * x).mean()
         
        return loss

    def __repr__(self):
        out_str = ', '.join([f'{x}' for x in self.fair_scaling_group_weights])
        out_str = '==> Fair_Loss_Scaler: ' + out_str
        return out_str

def compute_between_group_disparity(auc_list, overall_auc):
    return np.std(auc_list) / overall_auc, (np.max(auc_list)-np.min(auc_list)) / overall_auc

def compute_between_group_disparity_half(auc_list, overall_auc):
    return np.std(auc_list) / np.abs(overall_auc-0.5), (np.max(auc_list)-np.min(auc_list)) / np.abs(overall_auc-0.5)

def get_num_by_group(train_dataset_loader, n_group=3):
    samples_per_cls = [0]*n_group
    all_attrs = []
    for i, (input, target, attr) in enumerate(train_dataset_loader):
        attr_array = attr.detach().cpu().numpy().tolist()
        all_attrs = all_attrs + attr_array
        # for j in range(n_group):
        #     tmp = np.count_nonzero(attr_array == j)
        #     samples_per_cls[j] = samples_per_cls[j] + tmp
    all_attrs,samples_per_attr = np.unique(all_attrs, return_counts=True)

    return all_attrs, samples_per_attr

def get_num_by_group_(train_dataset_loader, n_group=7):
    samples_per_cls = [0]*n_group
    all_attrs = [ [] for _ in range(n_group)]
    for i, (input, input2, target, attr) in enumerate(train_dataset_loader):
        for i in range(len(attr)):
            one_attr = attr[i]
            attr_array = one_attr.detach().cpu().numpy().tolist()
            attr_array = [v for v in attr_array if not v < 0]
            all_attrs[i] = all_attrs[i] + attr_array
#     print(all_attrs)
    ret = []
    for i in range(len(all_attrs)):
        all_attrs_per_identity = all_attrs[i]
        _, samples_per_attr = np.unique(all_attrs_per_identity, return_counts=True)
        ret.append(samples_per_attr)
    
    return ret

def get_num_by_group_FA(train_dataset_loader, n_group=7):
    samples_per_cls = [0]*n_group
    all_attrs = [ [] for _ in range(n_group)]
    for i, (input, input2, target, attr_, attr) in enumerate(train_dataset_loader):
        for i in range(len(attr)):
            one_attr = attr[i]
            attr_array = one_attr.detach().cpu().numpy().tolist()
            attr_array = [v for v in attr_array if not v < 0]
            all_attrs[i] = all_attrs[i] + attr_array
#     print(all_attrs)
    ret = []
    for i in range(len(all_attrs)):
        all_attrs_per_identity = all_attrs[i]
        _, samples_per_attr = np.unique(all_attrs_per_identity, return_counts=True)
        ret.append(samples_per_attr)
    
    return ret

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class Rescaled_Softsign(nn.Module):
    def __init__(self, rescale=1.):
        super().__init__()
        self.rescale = rescale
        self.acti_func = nn.Softsign()

    def forward(self, x):
        y = self.acti_func(x)
        y = y*self.rescale
        return y

class Rescaled_Sigmoid(nn.Module):
    def __init__(self, rescale=1.):
        super().__init__()
        self.rescale = rescale
        self.acti_func = nn.Sigmoid()

    def forward(self, x):
        y = self.acti_func(x)-0.5
        y = y*self.rescale
        return y

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def reward_function(delta_err, reward_type='exp_delta', alpha=1.0):
    cur_reward = 0
    if reward_type == 'delta':
        cur_reward = alpha*max(delta_err, 0)
    elif reward_type == 'exp_delta':
        cur_reward = max(np.exp(alpha*delta_err)-1, 0)
    elif reward_type == 'log_delta':
        cur_reward = max(np.log(alpha*delta_err + 1), 0)
    elif reward_type == 'sqr_delta':
        cur_reward = max(alpha*delta_err, 0)**2
    return cur_reward

def reward_function_(y_hat, y_gt, reward_type='mae', eps=1e-6):
    cur_reward = 0
    if reward_type == 'mae':
        cur_reward = -torch.log(torch.abs(y_hat-y_gt)+eps).mean().item()
    elif reward_type == 'ce':
        cur_reward = F.binary_cross_entropy(y_hat, y_gt).item()
    return cur_reward

class General_Logistic(nn.Module):
    def __init__(self, min_val=-38.0, max_val=26.0):
        super().__init__()
        self.min_nml_val = min_val
        self.max_nml_val = max_val

        self.A = self.min_nml_val
        self.K = self.max_nml_val
        self.C = 1
        self.Q = 1
        self.B = 1

        self.nu = np.log2(self.C+self.Q*np.exp(1))/np.log2((self.K-self.A)/(-self.A))
        
    def forward(self, x):
        out = self.A + (self.K-self.A)/(self.C+self.Q*torch.exp( -self.B * x ))**(1/self.nu)
        return out

class Attribute_Grouped_Normalizer(nn.Module):
    def __init__(self, num_attr=0, dim=0, mus=None, sigmas=None, momentum=0.9):
        super().__init__()
        self.num_attr = num_attr
        self.dim=0
        self.mus = mus
        self.sigmas = sigmas
        self.eps = 1e-6
        self.momentum = momentum
        
    def forward(self, x, attr):
        if self.mus is None:
            self.mus = []
            for i in range(self.num_attr):
                self.mus.append(torch.zeros(x.shape[1]).type(x.type()))
        if self.sigmas is None:
            self.sigmas = []
            for i in range(self.num_attr):
                self.sigmas.append(torch.ones(x.shape[1]).type(x.type()))
        for idx in range(x.shape[0]):
            x[idx,:] = (x[idx,:] - self.mus[attr[idx]])/(self.sigmas[attr[idx]] + self.eps)
        
        return x

    def update_mus_sigmas(self, mus, sigmas):
        if self.momentum >= 0 and self.momentum < 1:
            for i in range(self.num_attr):
                self.mus[i] = self.momentum*self.mus[i] + (1-self.momentum)*mus[i]
                self.sigmas[i] = sigmas[i]

    def __repr__(self):
        if self.mus is not None and self.sigmas is not None:
            out_str = ', '.join([f'G{i}: ({torch.mean(m).item():f}, {torch.mean(s).item():f})' for i, (m, s) in enumerate(zip(self.mus, self.sigmas))])
        else:
            out_str = 'Attribute-Grouped Normalizer is not initialized yet.'
        return out_str

def forward_model_with_fin(model, data, attr):
    feat = model[0](data)
    if type(model[1]).__name__ != 'Fair_Identity_Normalizer' and type(model[1]).__name__ != 'Fair_Identity_Normalizer_':
        nml_feat = model[1](feat)
    else:
        nml_feat = model[1](feat, attr)
    logit = model[2](nml_feat)
    return logit, feat

def forward_model_with_fin_(model, data, attr):
    feat = data
    logit_feat = None
    for i, layer in enumerate(model.children()):
        if not type(layer).__name__.startswith('Fair_Identity_Normalizer'):
            feat = layer(feat)
        else:
            feat = layer(feat, attr)
        if i == len(model)-1:
            logit_feat = feat.detach().clone()
    logit = feat

    return logit, logit_feat

class Fair_Scaler(nn.Module):
    def __init__(self, beta=.9, bias=0, metric_scores=None):
        self.beta = beta
        self.bias = bias

    def forward(self, attr, metric_scores):
        metric_scores_ = metric_scores.clone().type_as(attr)
        weights = (1-self.beta) / (1-self.beta**(metric_scores_-self.bias))
        instance_weights = weights[attr]

        return instance_weights

class Fair_Identity_Normalizer_3D(nn.Module):
    def __init__(self, num_attr=0, dims=None, mu=0.001, sigma=0.1, momentum=0, test=False):
        super().__init__()
        self.num_attr = num_attr
        self.dims = dims

        
        self.mus = nn.Parameter(torch.randn(self.num_attr, self.dims[0], self.dims[1], self.dims[2])*mu)
        self.sigmas = nn.Parameter(torch.randn(self.num_attr, self.dims[0], self.dims[1], self.dims[2])*sigma)
        # self.sigma_coefs = nn.Parameter(torch.randn(self.num_attr, self.dim)*sigma)
        if test:
            self.sigmas = nn.Parameter(torch.ones(self.num_attr, self.dims[0], self.dims[1], self.dims[2])*sigma)
        self.eps = 1e-6
        self.momentum = momentum


    def forward(self, x, attr):
        x_clone = x.clone()
        for idx in range(x.shape[0]):
            x[idx,:,:,:] = (x[idx,:,:,:] - self.mus[attr[idx],:,:,:])/( torch.log(1+torch.exp(self.sigmas[attr[idx],:,:,:])) + self.eps)
        x = (1-self.momentum)*x + self.momentum*x_clone

        return x

    def __repr__(self):
        if self.mus is not None and self.sigmas is not None:
            sigma = torch.log(1+torch.exp(self.sigmas))
            sigma = torch.mean(sigma, dim=[1,2,3])
            mu = torch.mean(self.mus, dim=[1,2,3])
            out_str = ', '.join([f'G{i}: ({mu[i].item():f}, {sigma[i].item():f})' for i in range(mu.shape[0])])
        else:
            out_str = 'Attribute-Grouped Normalizer is not initialized yet.'
        return out_str

class Fair_Identity_Normalizer_Single(nn.Module):
    def __init__(self, num_attr=0, dim=0, mu=0.001, sigma=1.0, momentum=0, test=False):
        super().__init__()
        self.num_attr = num_attr
        self.dim = dim

        self.mus = nn.Parameter(torch.randn(self.num_attr)*mu)
        self.sigmas = nn.Parameter(torch.ones(self.num_attr)*sigma)

        if test:
            self.sigmas = nn.Parameter(torch.ones(self.num_attr)*sigma)
        self.eps = 1e-6
        self.momentum = momentum


    def forward(self, x, attr):
        x_clone = x.clone()
        for idx in range(x.shape[0]):
            x[idx,:] = (x[idx,:] - self.mus[attr[idx]])/( torch.log(1+torch.exp(self.sigmas[attr[idx]])) + self.eps)
        x = (1-self.momentum)*x + self.momentum*x_clone

        return x

    def __repr__(self):
        if self.mus is not None and self.sigmas is not None:
            sigma = torch.log(1+torch.exp(self.sigmas))
            mu = self.mus
            out_str = ', '.join([f'G{i}: ({mu[i].item():f}, {sigma[i].item():f})' for i in range(mu.shape[0])])
        else:
            out_str = 'Attribute-Grouped Normalizer is not initialized yet.'
        return out_str

class Fair_Identity_Normalizer(nn.Module):
    def __init__(self, num_attr=0, dim=0, mu=0.001, sigma=0.1, momentum=0, test=False):
        super().__init__()
        self.num_attr = num_attr
        self.dim = dim

        self.mus = nn.Parameter(torch.randn(self.num_attr, self.dim)*mu)
        self.sigmas = nn.Parameter(torch.randn(self.num_attr, self.dim)*sigma)
        
        if test:
            self.sigmas = nn.Parameter(torch.ones(self.num_attr, self.dim)*sigma)
        self.eps = 1e-6
        self.momentum = momentum


    def forward(self, x, attr):
        x_clone = x.clone()
        for idx in range(x.shape[0]):
            x[idx,:] = (x[idx,:] - self.mus[attr[idx], :])/( torch.log(1+torch.exp(self.sigmas[attr[idx], :])) + self.eps)
        x = (1-self.momentum)*x + self.momentum*x_clone

        return x

    def __repr__(self):
        if self.mus is not None and self.sigmas is not None:
            sigma = torch.log(1+torch.exp(self.sigmas))
            sigma = torch.mean(sigma, dim=1)
            mu = torch.mean(self.mus, dim=1)
            out_str = ', '.join([f'G{i}: ({mu[i].item():f}, {sigma[i].item():f})' for i in range(mu.shape[0])])
        else:
            out_str = 'Attribute-Grouped Normalizer is not initialized yet.'
        return out_str

class Fair_Identity_Normalizer_1D(nn.Module):
    def __init__(self, num_attr=0, dim=0, mu=0.001, sigma=0.1, sigma_coefs=None, momentum=0, test=False):
        super().__init__()
        self.num_attr = num_attr
        self.dim = dim
        self.mus = nn.Parameter(torch.randn(self.num_attr, self.dim)*mu)
        self.sigmas = nn.Parameter(torch.randn(self.num_attr, self.dim)*sigma)
        if test:
            self.sigmas = nn.Parameter(torch.ones(self.num_attr, self.dim)*sigma)
        self.eps = 1e-6
        self.momentum = momentum


    def forward(self, x, attr):
        x_clone = x.clone()
        for idx in range(x.shape[0]):
            x[idx,:] = (x[idx,:] - self.mus[attr[idx], :])/( torch.log(1 + torch.exp(self.sigmas[attr[idx], :])) + self.eps)
        x = (1-self.momentum)*x + self.momentum*x_clone

        return x

    def __repr__(self):
        if self.mus is not None and self.sigmas is not None:
            sigma = torch.log(1+torch.exp(self.sigmas))
            sigma = torch.mean(sigma, dim=1)
            mu = torch.mean(self.mus, dim=1)
            out_str = ', '.join([f'G{i}: ({mu[i].item():f}, {sigma[i].item():f})' for i in range(mu.shape[0])])
        else:
            out_str = 'Attribute-Grouped Normalizer is not initialized yet.'
        return out_str

class Fair_Identity_Normalizer_1D_(nn.Module):
    def __init__(self, num_attr=0, dim=0, mu=0.001, sigma=0.1, sigma_coefs=[1.,1.,1.], momentum=0, test=False):
        super().__init__()
        self.num_attr = num_attr
        self.dim = dim

        self.mus = nn.Parameter(torch.randn(self.num_attr, self.dim)*mu)
        self.sigmas = nn.Parameter(torch.randn(self.num_attr, self.dim)*sigma)
        if test:
            self.sigmas = nn.Parameter(torch.ones(self.num_attr, self.dim)*sigma)
        self.eps = 1e-6
        self.momentum = momentum


    def forward(self, x, attr):
        x_clone = x.clone()
        for idx in range(x.shape[0]):
            x[idx,:] = (x[idx,:] - self.mus[attr[idx], :])/( torch.log(1 + torch.exp(self.sigmas[attr[idx], :])) + self.eps)
            
        x = (1-self.momentum)*x + self.momentum*x_clone

        return x

    def __repr__(self):
        if self.mus is not None and self.sigmas is not None:
            sigma = torch.log(1+torch.exp(self.sigmas))
            sigma = torch.mean(sigma, dim=1)
            mu = torch.mean(self.mus, dim=1)
            out_str = ', '.join([f'G{i}: ({mu[i].item():f}, {sigma[i].item():f})' for i in range(mu.shape[0])])
        else:
            out_str = 'Attribute-Grouped Normalizer is not initialized yet.'
        return out_str

class Learnable_BatchNorm1d(nn.Module):
    def __init__(self, dim=0, mu=0, sigma=0.1, momentum=0.9):
        super().__init__()
        self.dim = dim
        self.mus = nn.Parameter(torch.ones(1, self.dim)*mu)
        self.sigmas = nn.Parameter(torch.ones(1, self.dim)*sigma)
        self.eps = 1e-6
        self.momentum = momentum

    def forward(self, x):
        for idx in range(x.shape[0]):
            x = (x - self.mus)/( torch.log(1+torch.exp(self.sigmas)) + self.eps)
        
        return x

    def __repr__(self):
        if self.mus is not None and self.sigmas is not None:
            sigma = torch.log(1+torch.exp(self.sigmas))
            sigma = torch.mean(sigma)
            mu = torch.mean(self.mus)
            out_str = f'Learnable BatchNorm: ({mu:f}, {sigma:f})'
        else:
            out_str = 'Attribute-Grouped Normalizer is not initialized yet.'
        return out_str

class MD_Mapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = 52
        self.out_features = 1
        
        weight = [0.010346387,0.010663622,0.010032727,0.007129987,0.014017274,0.018062957,0.018842243,0.016647837,0.015124109,0.011389459,0.014017678,0.022160035,0.02378898,0.02383174,0.02191793,0.019983033,0.0159671,0.0115242,0.007463015,0.018917531,0.023298082,0.02881147,0.027520778,0.025385285,0.023138773,0.016495131,0.008567998,0.017318338,0.028689633,0.02881154,0.028483851,0.025037148,0.023584995,0.016130119,0.015494349,0.024661184,0.028129123,0.028682529,0.026372951,0.024033034,-0.001105303,0.016997128,0.01889403,0.023627078,0.024890497,0.023402898,0.0218989,0.017713769,0.015848428,0.018916324,0.018597527,0.019021584]
        bias = 0.000592563
        self.weight = torch.nn.Parameter(torch.tensor(weight), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.tensor(bias), requires_grad=False)

    def forward(self, input):
        assert input.shape[1] == self.in_features
        output = input @ self.weight.t() + self.bias
        return output

class LogisticRegression(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear = torch.nn.Linear(input_dim, output_dim, bias=True)
     def forward(self, x):
         outputs = torch.sigmoid(self.linear(x))
         return outputs

class ConvNet_3D(nn.Module):
    def __init__(self, width=200, height=200, depth=200, out_dim=1, include_final=True):
        super().__init__()

        self.include_final = include_final
        
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2)  
        self.bn1 = nn.BatchNorm3d(64)
        
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        self.bn2 = nn.BatchNorm3d(64)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2)
        self.bn3 = nn.BatchNorm3d(128)
        
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=2)
        self.bn4 = nn.BatchNorm3d(256)
        
        self.gap = nn.AdaptiveAvgPool3d(1)
        
        self.fc1 = nn.Linear(256, 512)
        self.drop = nn.Dropout(0.3)
        
        if self.include_final:
            self.fc2 = nn.Linear(512, out_dim)
        else:
            self.fc2 = nn.Identity()
        
    def forward(self, x):
        x = F.relu(self.bn1(self.pool1(self.conv1(x))))
        x = F.relu(self.bn2(self.pool2(self.conv2(x))))
        x = F.relu(self.bn3(self.pool3(self.conv3(x))))
        x = F.relu(self.bn4(self.pool4(self.conv4(x))))
        
        x = torch.squeeze( self.gap(x), (2, 3, 4) )
        
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        
        x = self.fc2(x)
        
        return x

def create_model(model_type='efficientnet', in_dim=1, out_dim=1, use_pretrained=True, include_final=True):
    backbone = None
    if model_type == 'vit':
        backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        backbone.conv_proj = nn.Conv2d(in_dim, 768, kernel_size=(16, 16), stride=(16, 16))
        if include_final:
            backbone.heads[0] = nn.Linear(in_features=768, out_features=out_dim, bias=True)
        else:
            backbone.heads[0] = nn.Identity()
    elif model_type == 'efficientnet':
        load_weights = None
        if use_pretrained:
            load_weights = EfficientNet_B1_Weights.IMAGENET1K_V2
        backbone = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V2)
        if in_dim != 3:
            backbone.features[0][0] = nn.Conv2d(in_dim, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        if include_final:
            backbone.classifier[1] = nn.Linear(in_features=1280, out_features=out_dim, bias=False)
        else:
            backbone.classifier[1] = nn.Identity()
    elif model_type == 'efficientnet_v2':
        backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        if include_final:
            backbone.classifier[1] = nn.Linear(in_features=1280, out_features=out_dim, bias=True)
        else:
            backbone.classifier[1] = nn.Identity()
    elif model_type == 'resnet':
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        if in_dim != 3:
            backbone.conv1 = nn.Conv2d(in_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if include_final:
            backbone.fc = nn.Linear(in_features=2048, out_features=out_dim, bias=True)
        else:
            backbone.fc = nn.Identity()
    elif model_type == 'swin':
        backbone = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        if in_dim != 3:
            backbone.features[0][0] = nn.Conv2d(in_dim, 128, kernel_size=(4, 4), stride=(4, 4), padding=(1, 1), bias=False)
        backbone.head = nn.Linear(in_features=1024, out_features=out_dim, bias=True)
    elif model_type == 'vgg':
        backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        if in_dim != 3:
            backbone.features[0] = nn.Conv2d(in_dim, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        if include_final:
            backbone.classifier[6] = nn.Linear(in_features=4096, out_features=out_dim, bias=True)
        else:
            backbone.classifier[6] = nn.Identity()
    elif model_type == 'resnext':
        backbone = resnext101_64x4d(weights=ResNeXt101_64X4D_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Linear(in_features=2048, out_features=out_dim, bias=True)
    elif model_type == 'wideresnet':
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        backbone.fc = nn.Linear(in_features=2048, out_features=out_dim, bias=True)
    elif model_type == 'convnext':
        backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        if in_dim != 3:
            backbone.features[0][0] = nn.Conv2d(in_dim, 96, kernel_size=(4, 4), stride=(4, 4), padding=(1, 1), bias=False)
        backbone.classifier[2] = nn.Linear(in_features=768, out_features=out_dim, bias=True)
    elif model_type == 'densenet':
        backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        if in_dim != 3:
            backbone.features.conv0 = nn.Conv2d(in_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        backbone.classifier = nn.Linear(in_features=1024, out_features=out_dim, bias=True)
    return backbone

class Model_With_Time(nn.Module):
    def __init__(self, encoder=None, bias=True):
        super(Model_With_Time, self).__init__()
        self.encoder = encoder
        self.bias = bias
        self.classifier = nn.Linear(in_features=2, out_features=1, bias=self.bias)

    def forward(self, x, t):
        x_feat = self.encoder(x)
        x_feat = torch.cat((x_feat, t), dim=1)
        x_out = self.classifier(x_feat)
        return x_out

class OphBackbone_concat(nn.Module):
    def __init__(self, model_type='efficientnet', in_dim=1):
        super(OphBackbone, self).__init__()
        self.model_type = model_type
        self.in_dim = in_dim
        self.unit_feat_dim = 1280

        encoders = []
        for i in range(self.in_dim):
            cur_encoder = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V2)
            cur_encoder.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            cur_encoder.classifier[1] = nn.Linear(in_features=1280, out_features=1, bias=False)

            encoders.append(cur_encoder)
        self.encoders = nn.ModuleList(encoders)
        
        self.linear = nn.Linear(in_features=self.unit_feat_dim*self.in_dim, out_features=1, bias=True)

    def forward(self, x):
        x_out = []
        for i, l in enumerate(self.encoders):
            x_out.append(self.encoders[i](x[:,i:i+1,:,:]))
        x_out = torch.cat(x_out, dim=1)
        y = self.linear(x_out)
        return y

class OphBackbone(nn.Module):
    def __init__(self, model_type='efficientnet', in_dim=1, coef=1.):
        super(OphBackbone, self).__init__()
        self.model_type = model_type
        self.in_dim = in_dim
        self.unit_feat_dim = 1280
        self.coefs = [1., coef]

        encoders = []
        for i in range(self.in_dim):
            
            cur_encoder = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V2)
            cur_encoder.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            cur_encoder.classifier[1] = nn.Linear(in_features=1280, out_features=1, bias=False)
            
            encoders.append(cur_encoder)
        self.encoders = nn.ModuleList(encoders)

    def forward(self, x):
        x_out = None
        for i, l in enumerate(self.encoders):
            if x_out is None:
                x_out = self.coefs[i] * self.encoders[i](x[:,i:i+1,:,:])
            else:
                x_out += self.coefs[i] * self.encoders[i](x[:,i:i+1,:,:])
        y = x_out
        return y

class OphBackbone_(nn.Module):
    def __init__(self, model_type='efficientnet', in_dim=1):
        super(OphBackbone_, self).__init__()
        self.model_type = model_type
        self.in_dim = in_dim
        self.unit_feat_dim = 1280

        encoders = []
        for i in range(self.in_dim):
            cur_encoder = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            cur_encoder.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            cur_encoder.classifier[1] = nn.Linear(in_features=1280, out_features=1, bias=True)
            encoders.append(cur_encoder)
        self.encoders = nn.ModuleList(encoders)

    def forward(self, x):
        x_0 = self.encoders[0](x[:,0:0+1,:,:])
        x_1 = self.encoders[1](x[:,1:1+1,:,:])
        y = x_0 + x_1/(1+torch.abs(x_0))
        return y

class OphBackbone_Multiply(nn.Module):
    def __init__(self, model_type='efficientnet', in_dim=1):
        super(OphBackbone_, self).__init__()
        self.model_type = model_type
        self.in_dim = in_dim
        self.unit_feat_dim = 1280

        encoders = []
        for i in range(self.in_dim):
            cur_encoder = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            cur_encoder.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            cur_encoder.classifier[1] = nn.Linear(in_features=1280, out_features=1, bias=True)
            encoders.append(cur_encoder)
        self.encoders = nn.ModuleList(encoders)

    def forward(self, x):
        x_out = None
        for i, l in enumerate(self.encoders):
            if x_out is None:
                x_out = self.encoders[i](x[:,i:i+1,:,:])
            else:
                x_out *= self.encoders[i](x[:,i:i+1,:,:])
        y = x_out

        return y

class GlauClassifier(nn.Module):
    def __init__(self, ):
        super(GlauClassifier, self).__init__(model_type='efficientnet', in_dim=1, out_dim=1)
        self.model_type = model_type
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rnflt_encoder = create_model_(model_type=model_type, in_dim=in_dim, out_dim=out_dim)
        out_feat = -1
        if model_type == 'efficientnet':
            out_feat = 1280
        self.tds_encoder = nn.Sequential(nn.Linear(in_features=52, out_features=128, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=128, out_features=512, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=512, out_features=out_feat, bias=False))
        self.classifier = nn.Linear(in_features=out_feat*2, out_features=out_dim, bias=True)

    def forward(self, x, tds):
        rnflt_feat = self.rnflt_encoder(x)
        tds_feat = self.tds_encoder(tds)
        in_feat = torch.cat((rnflt_feat,tds_feat))
        y_hat = self.classifier(in_feat)
        return y_hat


def get_optimizer(optimizer_name, parameters, lr=0.001, **kwargs):
    """
    Returns the optimizer object based on the optimizer name and acceptable parameters.

    Args:
    - optimizer_name (str): Name of the optimizer (e.g., 'sgd', 'adam', etc.).
    - parameters: Model parameters to optimize.
    - lr (float, optional): Learning rate. Default is 0.001.
    - **kwargs: Additional keyword arguments specific to each optimizer.

    Returns:
    - torch.optim.Optimizer: The optimizer object.
    """
    # Define acceptable parameters for each optimizer
    optimizer_params = {
        'sgd': {'momentum', 'weight_decay', 'dampening', 'nesterov'},
        'adam': {'betas', 'eps', 'weight_decay', 'amsgrad'},
        'adamw': {'betas', 'eps', 'weight_decay', 'amsgrad'},
        'adagrad': {'lr_decay', 'weight_decay', 'eps'},
        'rmsprop': {'alpha', 'eps', 'weight_decay', 'momentum', 'centered'},
        'adadelta': {'rho', 'eps', 'weight_decay'},
        'adamax': {'betas', 'eps', 'weight_decay'},
        'asgd': {'lambd', 'alpha', 't0', 'weight_decay'}
    }

    optimizer_name = optimizer_name.lower()
    if optimizer_name not in optimizer_params:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Filter kwargs based on the optimizer's acceptable parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in optimizer_params[optimizer_name]}

    # Optimizer class mapping
    optimizers = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'adagrad': optim.Adagrad,
        'rmsprop': optim.RMSprop,
        'adadelta': optim.Adadelta,
        'adamax': optim.Adamax,
        'asgd': optim.ASGD
    }

    return optimizers[optimizer_name](parameters, lr=lr, **filtered_kwargs)


def get_scheduler(scheduler_name, optimizer, **kwargs):
    """
    Returns a learning rate scheduler object based on the scheduler name and its acceptable parameters.

    Args:
    - scheduler_name (str): Name of the scheduler (e.g., 'step_lr', 'exponential_lr').
    - optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
    - **kwargs: Additional keyword arguments specific to each scheduler.

    Returns:
    - torch.optim.lr_scheduler: The scheduler object.
    """
    # Define acceptable parameters for each scheduler
    scheduler_params = {
        'step_lr': {'step_size', 'gamma', 'last_epoch'},
        'multi_step_lr': {'milestones', 'gamma', 'last_epoch'},
        'exponential_lr': {'gamma', 'last_epoch'},
        'cosine_annealing_lr': {'T_max', 'eta_min', 'last_epoch'},
        'reduce_lr_on_plateau': {'mode', 'factor', 'patience', 'threshold', 'threshold_mode', 'cooldown', 'min_lr',
                                 'eps', 'verbose'},
        'cyclic_lr': {'base_lr', 'max_lr', 'step_size_up', 'step_size_down', 'mode', 'gamma', 'scale_fn', 'scale_mode',
                      'cycle_momentum', 'base_momentum', 'max_momentum', 'last_epoch'},
        'one_cycle_lr': {'max_lr', 'total_steps', 'epochs', 'steps_per_epoch', 'pct_start', 'anneal_strategy',
                         'cycle_momentum', 'base_momentum', 'max_momentum', 'div_factor', 'final_div_factor',
                         'last_epoch'}
    }

    if scheduler_name not in scheduler_params:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    # Filter kwargs based on the scheduler's acceptable parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in scheduler_params[scheduler_name]}

    # Scheduler class mapping
    schedulers = {
        'step_lr': lr_scheduler.StepLR,
        'multi_step_lr': lr_scheduler.MultiStepLR,
        'exponential_lr': lr_scheduler.ExponentialLR,
        'cosine_annealing_lr': lr_scheduler.CosineAnnealingLR,
        'reduce_lr_on_plateau': lr_scheduler.ReduceLROnPlateau,
        'cyclic_lr': lr_scheduler.CyclicLR,
        'one_cycle_lr': lr_scheduler.OneCycleLR
    }

    return schedulers[scheduler_name](optimizer, **filtered_kwargs)
