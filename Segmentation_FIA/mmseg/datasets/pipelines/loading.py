# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import os.path as osp

import mmcv
import numpy as np
import pdb
from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        
        # print('+++++', results)
        
        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        # img_bytes = self.file_client.get(filename)
        
        # print(filename)
        # img = mmcv.imfrombytes(
            # img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            
        data = np.load(filename, allow_pickle=True)
        
        if(results['fundus_split'] == "fundus_slo"): # target
            img = data["fundus_slo"]
        elif(results['fundus_split'] == "fundus_oct"): # source
            img = data["fundus_oct"]
        
        # print("loading 1", img.shape)
        if self.to_float32:
            img = img.astype(np.float32)
        
        # print("img.shape", img.shape, np.max(img), np.min(img))
        img = np.expand_dims(img, axis=-1)
        img = np.repeat(img, 3, axis=2)
        
        # print("loading 2", img.shape)

        results['filename'] = filename
        # results['fundus_split'] = results['fundus_split']
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

hashmap = {-1:1, -2:2, 0:0}

@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        # print('-------', results)
#         if results.get('seg_prefix', None) is not None:
#             filename = osp.join(results['seg_prefix'],
#                                 results['ann_info']['seg_map'])
#         else:
#             filename = results['ann_info']['seg_map']
            
            
#         img_bytes = self.file_client.get(filename)
#         gt_semantic_seg = mmcv.imfrombytes(
#             img_bytes, flag='unchanged',
#             backend=self.imdecode_backend).squeeze().astype(np.uint8)
#         # modify if custom classes
#         if results.get('label_map', None) is not None:
#             for old_id, new_id in results['label_map'].items():
#                 gt_semantic_seg[gt_semantic_seg == old_id] = new_id
#         # reduce zero_label
#         if self.reduce_zero_label:
#             # avoid using underflow conversion
#             gt_semantic_seg[gt_semantic_seg == 0] = 255
#             gt_semantic_seg = gt_semantic_seg - 1
#             gt_semantic_seg[gt_semantic_seg == 254] = 255

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        # img_bytes = self.file_client.get(filename)
        
        # print(filename)
        # img = mmcv.imfrombytes(
            # img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            
        data = np.load(filename, allow_pickle=True)
        
        # gt_semantic_seg = np.zeros((512,512))
        
        if(results['fundus_split'] == "fundus_slo"): # target
            gt_semantic_seg = data["slo_disc_cup"]
            
            # print("slo_disc_cup ----", np.unique(data["slo_disc_cup"]))
            
        elif(results['fundus_split'] == "fundus_oct"): # source
            gt_semantic_seg = data["oct_disc_cup"]
            
            # print("oct_disc_cup ----", np.unique(data["oct_disc_cup"]))
        
        # attr_label = 
        # print(attr_label)
        # print(results['str_attr_label'])
        attr_label = data[results['str_attr_label']].item()
        
        attr_to_race = {2: 0, 3: 1, 7:2}
        # print(filename, attr_label)
        if results['str_attr_label'] == 'race':
            attr_label = attr_to_race[attr_label]
        
        attr_to_hispanic = {0: 0, -1: 0, 1:1}
        if results['str_attr_label'] == 'hispanic':
            attr_label = attr_to_hispanic[attr_label]
            
        # print("---", attr_label)
        
        # print("before ----", np.unique(gt_semantic_seg))
        for i in range(len(gt_semantic_seg)):
            label = gt_semantic_seg[i]
            for k in sorted(hashmap.keys()):
                label[label == k] = hashmap[k]
            gt_semantic_seg[i] = label
        # print("after ----", np.unique(gt_semantic_seg))
            
        results['gt_semantic_seg'] = gt_semantic_seg
        results['attr_label'] = attr_label
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str
