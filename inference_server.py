"""Flask based API Serving
"""
import argparse
import os

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader,DistributedSampler
import cv2
import numpy as np

import mmcv
from mmcv import DictAction
from mmdet.apis import inference_detector
from mmpose.apis import (inference_top_down_pose_model)
from mmaction.apis import inference_recognizer_i,init_recognizer
from mmaction.datasets.pipelines import UniformSampleFrames
from mmdeploy.apis import build_task_processor
from mmdeploy.utils.config_utils import load_config

import boto3
from flask import Flask, request

app = Flask(__name__)

def parse_args():
    """Generate Arguments

    Returns:
        argparse.Namespace : arguments
    """
    parser = argparse.ArgumentParser(description='Quantom Short Video Demo')

    parser.add_argument(
        '--det-config',
        default='../mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py',
        help='human det config file path (from mmdetection)')

    parser.add_argument(
        '--det-deploy-config',
        default='../mmdeploy/configs/mmdet/detection/detection_tensorrt-fp16_dynamic-320x320-1344x1344.py',
        help='human det deploy config file path (from mmdetection)')

    parser.add_argument(
        '--det-checkpoint',
        type=str, nargs='+',
        default=('./work_dirs/local/faster_rcnn_r50_fpn_1x_coco-person.engine'),
        help='human det checkpoint file/url')

    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.5,
        help='the threshold of action prediction score')

    parser.add_argument(
        '--pose-config',
        default='../mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py',
        help='human pose estimation config file path (from mmpose)')

    parser.add_argument(
    '--pose-deploy-config',
    default='../mmdeploy/configs/mmpose/pose-detection_tensorrt-fp16_static-256x192.py',
    help='human det deploy config file path (from mmdetection)')

    parser.add_argument(
        '--pose-checkpoint',
        type=str, nargs='+',
        default=('./work_dirs/local/hrnet_w32_coco_256x192.engine'),
        help='human pose estimation checkpoint file/url')

    parser.add_argument(
        '--skeleton-config',
        default='configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_keypoint.py',
        help='skeleton-based action recognition config file path')

    parser.add_argument(
        '--skeleton-checkpoint',
        default='./work_dirs/local/slowonly_r50_u48_240e_ntu120_xsub_keypoint/epoch_1.pth',
        help='skeleton-based action recognition checkpoint file/url')
    
    parser.add_argument(
        '--skeleton-score-thr',
        type=float,
        default=0.4,
        help='the threshold of action prediction score')

    parser.add_argument(
        '--video_list',
        action='append',
        help='video folder path')

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")

    parser.add_argument(
        '--det-batch',
        default=1)

    parser.add_argument(
        '--pose-batch',
        default=1)

    parser.add_argument(
        '--action-batch',
        default=1)

    parser.add_argument(
        '--gpus',
        default=1)

    args = parser.parse_args()
    return args

def frame_extraction(video_path):
    """Extract frames given video_name.

    Args:
        video_name (str): The video_name.
    Returns:
        np.ndarray : frame image array
    """
    
    vid = cv2.VideoCapture(video_path)
    
    frames = []
    flag, frame = vid.read()
    cnt = 0
    
    while flag:
        frames.append(frame)
        cnt += 1

        flag, frame = vid.read()
    
    return np.asarray(frames)

def detection_inference_trt(args, frames, det_deploy_cfg,det_model,det_task_processor):
    """Object Detection TensorRT Inference

    Args:
        args (class): arguments
        frames (np.ndarray): frame image array
        det_deploy_cfg(str) : object detection tensorrt config file path
        det_model (class): object detection model
        det_task_processor(BaseTask) : A task processor

    Returns:
        list: object detection result list
    """
    results = []

    print('')
    print('Performing Human Detection for each frame')

    frame_num = len(frames)    

    prog_bar = mmcv.ProgressBar(frame_num/args.det_batch)

    for batch_idx in range(0,frame_num,args.det_batch): 
        result = inference_detector(det_model,frames[batch_idx:batch_idx+args.det_batch],det_deploy_cfg=det_deploy_cfg,det_task_processor=det_task_processor)
        # We only keep human detections with score larger than det_score_thr

        for _ in result:
            det = _[0][_[0][:,4]>=args.det_score_thr]
            results.append(det)

        prog_bar.update()

    print('')
    return results

def pose_inference_trt(args, frames, det_results, pose_deploy_cfg, pose_model,pose_task_processor):
    """Pose Estimation Inference

    Args:
        frames (np.ndarray): frmae image array
        det_results (list): object detection result list
        pose_deploy_cfg (str): pose estimation config file path
        pose_model (class): pose estimation model
        pose_task_processor (BaseTask) : A task processor

    Returns:
        list: pose estimation result list
    """
    ret = []
    prog_bar = mmcv.ProgressBar(len(frames))
    
    for f, d in zip(frames, det_results):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(pose_model,f, d,pose_deploy_cfg=pose_deploy_cfg, pose_task_processor=pose_task_processor,bbox_limit=args.pose_batch, format='xyxy')[0]
        ret.append(pose)
        prog_bar.update()

    print('')
    return ret

@app.route('/Inference',methods=['POST'])
def Run_Inference(videos):
    """Multi Processing Main Function

    Args:
        Inference_func (function): Inference Function
    """
    args = parse_args()
    args.gpus=torch.cuda.device_count()
    args.det_batch=48
    args.pose_batch=5
    args.action_batch=1
    #추후 수정 필요
    args.video_list = [f.filename for f in request.files.getlist("file")]

    mp.spawn(Inference,
            args=(args,),
            nprocs=args.gpus,
            join=True)
    
    #추후 수정 필요
    return 'OK'

def Inference(rank,args):
    """Inference Function

    Args:
        rank (int): _description_
        args (argparse.Namespace): arguments

    Returns:
        None
    """
    #추후 수정 필요
    s3 = boto3.client('s3')
    bucket_name = 'name_s3'

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    #Initialization of Process Group
    dist.init_process_group(backend='nccl',rank=rank,world_size=args.gpus)
    
    #Get Model Parameters
    det_deploy_cfg_path = args.det_deploy_config
    det_cfg_path = args.det_config
    pose_deploy_cfg_path = args.pose_deploy_config
    pose_cfg_path = args.pose_config

    det_deploy_cfg, det_cfg = load_config(det_deploy_cfg_path,det_cfg_path)
    pose_deploy_cfg, pose_cfg = load_config(pose_deploy_cfg_path,pose_cfg_path)
    skeleon_cfg = mmcv.Config.fromfile(args.skeleton_config)

    if args.cfg_options is not None:
        det_cfg.merge_from_dict(args.cfg_options)
        pose_cfg.merge_from_dict(args.cfg_options)
        skeleon_cfg.merge_from_dict(args.cfg_options)

    for component in skeleon_cfg.data.test.pipeline:
        if component['type'] == 'PoseNormalize': 
            component['mean'] = (w // 2, h // 2, .5)
            component['max_value'] = (w, h, 1.)

    is_cuda = (torch.cuda.is_available())
    assert is_cuda, ("We don't offer Cpu Mode, Please Run with Gpus")
    
    #Initialization of Object Detection and Pose Estimation and Action Recognition Model
    device = 'cuda:'+str(rank)

    det_task_processor = build_task_processor(det_cfg, det_deploy_cfg, device) 
    det_model = det_task_processor.init_backend_model([args.det_checkpoint])

    pose_task_processor = build_task_processor(pose_cfg, pose_deploy_cfg, device)
    pose_model = pose_task_processor.init_backend_model([args.pose_checkpoint])

    action_recognition_model = init_recognizer(skeleon_cfg, args.skeleton_checkpoint, device)

    #Initialization of Data Loader and Sampler
    #We Use Distirbuted Sampler Because Our Code is Running On Multi GPU
    sampler = DistributedSampler(
        args.video_list,
        num_replicas = args.gpus,
        rank=rank,
        shuffle=False
    )

    data_loader = DataLoader(
        args.video_list,
        batch_size = args.action_batch,       
        num_workers=0,
        sampler = sampler,
        shuffle = False,
        pin_memory = True,
    )

    for video_name in data_loader:
        video_name = video_name[0] 
        frames = frame_extraction(video_name)
        num_frame = len(frames)
        num_time_step = 48
        h, w, _ = frames[0].shape

        #Apply UniformSampling
        sampler = UniformSampleFrames(clip_len=num_time_step, num_clips=1, test_mode=True)
        results = dict(total_frames=num_frame, start_index=0)
        sampling_results = sampler(results)

        torch.cuda.empty_cache()

        #Get Object Detection and Pose Estimation Results
        det_results = detection_inference_trt(args,frames[sampling_results['frame_inds']],det_deploy_cfg,det_model,det_task_processor)
        pose_results = pose_inference_trt(args,frames[sampling_results['frame_inds']],det_results, pose_deploy_cfg, pose_model, pose_task_processor)

        persons = [len(x) for x in det_results]
        num_person = max(persons) if len(persons) else 0
        num_keypoint = 17

        #Generate Action Recognition Input Format
        fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_time_step)

        fake_anno['frame_inds'] = np.array(range(num_time_step))
        fake_anno['clip_len'] = sampling_results['clip_len']
        fake_anno['frame_interval'] = sampling_results['frame_interval']
        fake_anno['num_clips'] = sampling_results['num_clips']

        keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                            dtype=np.float16)
        keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                                dtype=np.float16)
                                
        for i, poses in enumerate(pose_results):
            for j, pose in enumerate(poses):
                pose = pose['keypoints']
                keypoint[j, i] = pose[:, :2]
                keypoint_score[j, i] = pose[:, 2]

        fake_anno['keypoint'] = keypoint
        fake_anno['keypoint_score'] = keypoint_score

        #Get Action Recognition Result
        result = inference_recognizer_i(action_recognition_model, fake_anno,samples_per_gpu=args.action_batch)[0][1]

        if result>=args.skeleton_score_thr:
            #추후 수정 필요
            #s3로 해당 비디오 및 데이터 전달
            target_video_path = video_name
            s3.upload_file(video_name,bucket_name,target_video_path)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080)