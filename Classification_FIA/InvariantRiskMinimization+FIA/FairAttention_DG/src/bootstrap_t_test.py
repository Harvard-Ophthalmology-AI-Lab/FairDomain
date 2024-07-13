import os
import numpy as np
import random
import pandas as pd
from sklearn.metrics import *
import scipy.stats as stats 
from sklearn.utils import *

import torch

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

def bootstrap(preds, gts, repeat_times=10, random_state=0, n_samples=100):
    bootstrap_aucs = []
    for i in range(repeat_times):
        # # Sample size for bootstrapping
        # sample_size = len(preds)

        # # Generating random indices for bootstrapping
        # random_indices = np.random.choice(len(preds), size=sample_size, replace=True)

        # # Extracting pairs using the generated indices
        # bootstrap_pairs_array1 = preds[random_indices]
        # bootstrap_pairs_array2 = gts[random_indices]

        bootstrap_pairs_array1, bootstrap_pairs_array2 = resample(preds, gts, replace=True, n_samples=n_samples) #, n_samples=sample_size, random_state=random_state

        overall_auc = compute_auc(bootstrap_pairs_array1, bootstrap_pairs_array2, num_classes=2)
        bootstrap_aucs.append(overall_auc)

    return bootstrap_aucs


input_npz_1 = 'results_cvpr_main/vit-b16_use_note/glaucoma_CLIP_vit-b16_1e-5_bz32_note_seed7680_auc0.6797/pred_gt_ep004.npz'
input_npz_2 = 'results_cvpr_main/vit-b16_use_note_race_fair/glaucoma_CLIP_vit-b16_1e-5_bz32_note_lambda1e-7_blur0.4_fairbz32_seed8567_auc0.7210/pred_gt_ep002.npz'
repeat_times = [100, 1000, 2000, 3000, 4000, 5000]
# n_samples = 2000


raw_data = np.load(input_npz_1)
# ['val_pred', 'val_gt', 'val_attr']
preds_1 = raw_data['val_pred']
gts_1 = raw_data['val_gt']

raw_data = np.load(input_npz_2)
preds_2 = raw_data['val_pred']
gts_2 = raw_data['val_gt']

random_state = random.randint(0, 1e+6)

print(f'random state: {random_state}')

for i in repeat_times:
    auc_1st_npz = bootstrap(preds_1, gts_1, repeat_times=i, n_samples=len(preds_1))
    auc_2nd_npz = bootstrap(preds_2, gts_2, repeat_times=i, n_samples=len(preds_1))
    # print(np.mean(auc_1st_npz), np.mean(auc_2nd_npz))
    result = stats.ttest_rel(auc_1st_npz, auc_2nd_npz) 
    print(f'repeat {i} times, p-value: {result.pvalue}')
    # print(result)