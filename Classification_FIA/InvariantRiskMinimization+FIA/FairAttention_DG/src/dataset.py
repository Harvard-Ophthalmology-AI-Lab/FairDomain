
from torch.utils.data import Dataset
import torch
import numpy as np
import os
from PIL import Image
import cv2
from torchvision import transforms

# from rnflt_plot import *


class RNFLT_Dataset_Artifact(Dataset):
    # subset: train | val | test | unmatch
    def __init__(self, data_path='./data/', subset='train', modality_type='rnflt', resolution=224, need_shift=True, transform=None):

        self.data_path = data_path
    
        self.modality_type = modality_type
        self.subset = subset
        self.transform = transform
        self.resolution = resolution
        self.need_shift = need_shift

        

        if self.modality_type == 'rnflt':
            self.pair_file_follow = []
            self.pair_file_base = []
       
           
            # self.rnflt_path_follow = os.path.join(self.data_path, 'rnflt_align')
            self.rnflt_path = os.path.join(self.data_path, 'output_rnflt_base_with_follow_artifact_corrected')
            
          
            self.file_list_rnflt= os.listdir(self.rnflt_path)
            
            if self.subset == 'train':
                self.file_list_rnflt = self.file_list_rnflt[:12000]
                
            else:
                self.file_list_rnflt = self.file_list_rnflt[12000:]

    def __len__(self):
        return len(self.file_list_rnflt)
        
    def crop_center(self, img, cropx, cropy):

        y,x = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)    
        return img[starty:starty+cropy,startx:startx+cropx]

    def crop_image(self, img, tol=-2):
        # img is 2D image data
        # tol  is tolerance
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]

    def interval_mapping(self, image, from_min, from_max, to_min, to_max):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
        from_range = from_max - from_min
        to_range = to_max - to_min
        scaled = np.array((image - from_min) / float(from_range), dtype=float)
        return to_min + (scaled * to_range)

    def __getitem__(self, item):

        if self.modality_type == 'rnflt':
            rnflt_file = self.file_list_rnflt[item]
            rnflt_path = os.path.join(self.rnflt_path, rnflt_file)
            rnflt_base_img = np.load(rnflt_path, allow_pickle=True)['rnflt_base_crop_artifact_correct']
            # rnflt_base_img = self.interval_mapping(rnflt_base_img, -2, 350, 0, 255)
            #rnflt_base_img = cv2.normalize(rnflt_base_img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            
            follow_img = np.load(rnflt_path, allow_pickle=True)['rnflt_follow_crop_artifact_correct']
            
            residual_map = np.load(rnflt_path, allow_pickle=True)['residual']
          

            return follow_img/350, rnflt_base_img/350, residual_map




class RNFLT_Dataset_Pred_Tds(Dataset):
    # subset: train | val | test | unmatch
    def __init__(self, data_path='./data/', subset='train', modality_type='rnflt', resolution=224, need_shift=True, transform=None):

        self.data_path = data_path
    
        self.modality_type = modality_type
        self.subset = subset
        self.transform = transform
        self.resolution = resolution
        self.need_shift = need_shift

        
        self.pair_file_follow = []
        self.pair_file_base = []
    
        
        # self.rnflt_path_follow = os.path.join(self.data_path, 'rnflt_align')
        self.rnflt_path = self.data_path
        
        
        self.file_list_rnflt = os.listdir(self.rnflt_path)
        # print(len( self.file_list_rnflt ))
        
        
        pids, dict_pid_fid = self.get_all_pids(self.file_list_rnflt)
        pids = np.array(pids)
        print(f"# of patients: {len(pids)}")
        
        train_file_lists = []
        test_file_lists = []
        train_pids = pids[:8500]
        test_pids = pids[8500:]
    
        for train_id in train_pids: 
            file_names = dict_pid_fid[train_id] 
            train_file_lists = train_file_lists + file_names
            # print(file_names)
        for test_id in test_pids: 
            file_names = dict_pid_fid[test_id] 
            test_file_lists = test_file_lists + file_names
        # print(len(train_file_lists)+ len(test_file_lists))
       
        # exit(1)

        # print(len(self.file_list_rnflt))
        
        
        if self.subset == 'train':
            self.file_list = train_file_lists
            
        else:
            self.file_list = test_file_lists

    def __len__(self):
        return len(self.file_list)
    
    def get_all_pids(self, all_files):
        pids = []
        dict_pid_fid = {}
        for i, f in enumerate(all_files):
            # raw_data = np.load(os.path.join(data_dir, f))
            # print(f)
            pid = f.split('_')[0]
            
            # pid = raw_data['pid'].item() 
            # pid = pid[:pid.find('_')]
            # print(pid)
            if pid not in dict_pid_fid:
                dict_pid_fid[pid] = [f]
            else:
                dict_pid_fid[pid].append(f)
            pids.append(pid)
        
        pids = list(dict_pid_fid.keys())
        return pids, dict_pid_fid

    def crop_center(self, img, cropx, cropy):

        y,x = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)    
        return img[starty:starty+cropy,startx:startx+cropx]

    def crop_image(self, img, tol=-2):
        # img is 2D image data
        # tol  is tolerance
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]

    def interval_mapping(self, image, from_min, from_max, to_min, to_max):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
        from_range = from_max - from_min
        to_range = to_max - to_min
        scaled = np.array((image - from_min) / float(from_range), dtype=float)
        return to_min + (scaled * to_range)

    def __getitem__(self, item):

        # rnflt_file = self.file_list_rnflt[item]
        rnflt_file = self.file_list[item]
        
        rnflt_path = os.path.join(self.rnflt_path, rnflt_file)
        npz_file = np.load(rnflt_path, allow_pickle=True)
        rnflt_img = npz_file['rnflt_follow'].astype(np.float32)
       
        td_values = npz_file['tds'].astype(np.float32)/38.
        # data_dir = npz_file['datadir']
        # print(data_dir)
        # print()
        
        if self.modality_type == 'rnflt':
            rnflt_img = np.clip(rnflt_img, -2, 350)
            # rnflt_img = rnflt_img - 2
            rnflt_img = rnflt_img[np.newaxis, :, :]
            # disk_rim_mask = (rnflt_img ==-1) + (rnflt_img ==-2)
            # disk_rim_mask = np.nonzero(disk_rim_mask)
            data_sample = rnflt_img.astype(np.float32)
            # data_sample = self.interval_mapping(data_sample, -2, 350, 0, 1)
            # data_sample = data_sample.astype(np.float32)
        elif self.modality_type == 'multi':
            rnflt_img = np.clip(rnflt_img, -2, 350)
            # rnflt_img = rnflt_img - 2
            rnflt_img = rnflt_img[np.newaxis, :, :]
            residual = npz_file['residual'].astype(np.float32)
            # residual = np.squeeze(residual, axis=0)
            # data_dir = npz_file['datadir']
            
            # data_sample = rnflt_img.astype(np.float32)
            data_sample = np.concatenate((rnflt_img, residual), axis=0)
            data_sample = data_sample.astype(np.float32)
        elif self.modality_type == 'residual':
            # rnflt_img = np.clip(rnflt_img, -2, 350)
            # # rnflt_img = rnflt_img - 2
            # rnflt_img = rnflt_img[np.newaxis, :, :]
            # disk_rim_mask = (rnflt_img ==-1) + (rnflt_img ==-2)
            # disk_rim_mask = np.nonzero(disk_rim_mask)
            residual = npz_file['residual'].astype(np.float32)
            residual = residual[np.newaxis, :, :] # if use min's model
            # data_dir = npz_file['datadir']
            residual = np.clip(residual, 0, 350)
            gaussian_smoothing = transforms.GaussianBlur(7, sigma=1)
            # print(residual.shape)
            residual = gaussian_smoothing(torch.tensor(residual))
            # print(residual.shape)
            # residual[disk_rim_mask] = 0
            # residual = self.interval_mapping(residual, 0, 350, 0, 1)

            data_sample = residual.numpy().astype(np.float32)
            # print(data_sample.shape)
            # exit(1)
            # data_sample = np.concatenate((rnflt_img, residual), axis=0)
            # data_sample = data_sample.astype(np.float32)
        # td_values = self.interval_mapping(td_values, -26, 38, 0, 1).astype(np.float32)
        return data_sample, td_values
        

if __name__ == "__main__":

    basefile_path = '/data/home/tiany/Datasets/min_residual/predisease_predict_VF_data_20000samples_with_residual'

    Dataset = RNFLT_Dataset_Pred_Tds(data_path=basefile_path, subset='train', modality_type='residual', resolution=224, need_shift=True)
    print(len(Dataset))
    for i in range(len(Dataset)):
        data = Dataset[i]
        
        exit(1)
