"""Action Recognition Single Gpu Inference
"""
from cProfile import label
import time
from tqdm import tqdm
import argparse
import os
import os.path as osp

import torch
import cv2
import numpy as np
import pickle
import pandas as pd

import mmcv
from mmcv import DictAction
from mmaction.apis import inference_recognizer, init_recognizer


def parse_args():
    """Generate Arguments

    Returns:
        argparse.Namespace : arguments
    """
    parser = argparse.ArgumentParser(description='Quantom Short Video Demo')
    parser.add_argument(
        '--skeleton-config',
        default='mmaction2/configs/skeleton/posec3d/dacon_hand_posec3d.py',
        help='skeleton-based action recognition config file path')
    parser.add_argument(
        '--skeleton-checkpoint',
        default='/home/sckim/Dataset/ckpt/dacon_posec3d/epoch_35.pth',
        help='skeleton-based action recognition checkpoint file/url')
    parser.add_argument(
        '--test-pkl', type=str, default='/home/sckim/Dataset/Competition/dacon_hand/test.pkl', help='test pkl path')
    parser.add_argument(
        '--out-path', type=str, default='/home/sckim/Dataset/Competition/dacon_hand/result_35_tta.csv', help='test pkl path')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.test_pkl, 'rb') as f:
        test_data = pickle.load(f)
        f.close()

    # Get clip_len, frame_interval and calculate center index of each clip
    config = mmcv.Config.fromfile(args.skeleton_config)
    config.merge_from_dict(args.cfg_options)
    action_recognition_model = init_recognizer(
        config, args.skeleton_checkpoint, args.device)

    anno = {}
    for data in tqdm(test_data):
        torch.cuda.empty_cache()

        # Get Start Time

        data['label'] = -1
        data['start_index'] = 0
        data['modality'] = 'Pose'

        # Get Action Recognition Result
        result = inference_recognizer(
            action_recognition_model, data, samples_per_gpu=1)[0][0]

        anno[data['frame_dir']] = result

    anno = sorted(anno.items())
    ids = [ann[0] for ann in anno]
    labels = [ann[1] for ann in anno]

    df = pd.DataFrame({
        'id': pd.Series(ids),
        'label': pd.Series(labels)
    })

    df.to_csv(args.out_path, index=False)


if __name__ == '__main__':
    main()
