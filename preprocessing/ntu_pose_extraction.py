# Copyright (c) OpenMMLab. All rights reserved.
"""동영상으로 부터 keypoint 데이터 셋(.pkl) 생성
   Modify : 하나의 동영상으로부터 pkl 생성 -> 전체 동영상으로부터 전체 pkl 생성(Train / Validation / Test 분할)
"""

import pickle
import abc
import argparse
from tqdm import tqdm
import os
import os.path as osp
import random as rd
import shutil
import string
import warnings
from collections import defaultdict

import cv2
import mmcv
import numpy as np

try:
    from mmdet.apis import inference_detector, init_detector
    from mmpose.apis import inference_top_down_pose_model, init_pose_model
except ImportError:
    warnings.warn(
        'Please install MMDet and MMPose for NTURGB+D pose extraction.'
    )  # noqa: E501

mmdet_root = '~/sckim/3.project/9.Action_Recognition/quantom/program/mmdetection'
mmpose_root = '~/sckim/3.project/9.Action_Recognition/quantom/program/mmpose'

args = abc.abstractproperty()
args.det_config = f'{mmdet_root}/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py'  # noqa: E501
args.det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'  # noqa: E501
args.det_score_thr = 0.5
args.pose_config = f'{mmpose_root}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py'  # noqa: E501
args.pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'  # noqa: E501
args.device = 'cuda:0'
args.skip_postproc = True

#Load Object Detection and Pose Estimation Models
detmodel = init_detector(args.det_config, args.det_checkpoint, args.device)
posemodel = init_pose_model(args.pose_config, args.pose_checkpoint,
                            args.device)

def gen_id(size=8):
    chars = string.ascii_uppercase + string.digits
    return ''.join(rd.choice(chars) for _ in range(size))

def extract_frame(video_path):
    dname = gen_id()
    os.makedirs(dname, exist_ok=True)
    frame_tmpl = osp.join(dname, 'img_{:05d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    while flag:
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()

    return frame_paths

def detection_inference(args, frame_paths):
    assert detmodel.CLASSES[0] == 'person', ('We require you to use a detector '
                                          'trained on COCO')
    results = []
    print('Performing Human Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(detmodel, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= args.det_score_thr]
        results.append(result)
        prog_bar.update()
    return results

def intersection(b0, b1):
    l, r = max(b0[0], b1[0]), min(b0[2], b1[2])
    u, d = max(b0[1], b1[1]), min(b0[3], b1[3])
    return max(0, r - l) * max(0, d - u)

def iou(b0, b1):
    i = intersection(b0, b1)
    u = area(b0) + area(b1) - i
    return i / u

def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])

def removedup(bbox):
    def inside(box0, box1, thre=0.8):
        return intersection(box0, box1) / area(box0) > thre

    num_bboxes = bbox.shape[0]
    if num_bboxes == 1 or num_bboxes == 0:
        return bbox
    valid = []
    for i in range(num_bboxes):
        flag = True
        for j in range(num_bboxes):
            if i != j and inside(bbox[i],
                                 bbox[j]) and bbox[i][4] <= bbox[j][4]:
                flag = False
                break
        if flag:
            valid.append(i)
    return bbox[valid]


def is_easy_example(det_results, num_person):
    threshold = 0.95

    def thre_bbox(bboxes, thre=threshold):
        shape = [sum(bbox[:, -1] > thre) for bbox in bboxes]
        ret = np.all(np.array(shape) == shape[0])
        return shape[0] if ret else -1

    if thre_bbox(det_results) == num_person:
        det_results = [x[x[..., -1] > 0.95] for x in det_results]
        return True, np.stack(det_results)
    return False, thre_bbox(det_results)

def bbox2tracklet(bbox):
    iou_thre = 0.6
    tracklet_id = -1
    tracklet_st_frame = {}
    tracklets = defaultdict(list)
    for t, box in enumerate(bbox):
        for idx in range(box.shape[0]):
            matched = False
            for tlet_id in range(tracklet_id, -1, -1):
                cond1 = iou(tracklets[tlet_id][-1][-1], box[idx]) >= iou_thre
                cond2 = (
                    t - tracklet_st_frame[tlet_id] - len(tracklets[tlet_id]) <
                    10)
                cond3 = tracklets[tlet_id][-1][0] != t
                if cond1 and cond2 and cond3:
                    matched = True
                    tracklets[tlet_id].append((t, box[idx]))
                    break
            if not matched:
                tracklet_id += 1
                tracklet_st_frame[tracklet_id] = t
                tracklets[tracklet_id].append((t, box[idx]))
    return tracklets

def drop_tracklet(tracklet):
    tracklet = {k: v for k, v in tracklet.items() if len(v) > 5}

    def meanarea(track):
        boxes = np.stack([x[1] for x in track]).astype(np.float32)
        areas = (boxes[..., 2] - boxes[..., 0]) * (
            boxes[..., 3] - boxes[..., 1])
        return np.mean(areas)

    tracklet = {k: v for k, v in tracklet.items() if meanarea(v) > 5000}
    return tracklet

def distance_tracklet(tracklet):
    dists = {}
    for k, v in tracklet.items():
        bboxes = np.stack([x[1] for x in v])
        c_x = (bboxes[..., 2] + bboxes[..., 0]) / 2.
        c_y = (bboxes[..., 3] + bboxes[..., 1]) / 2.
        c_x -= 480
        c_y -= 270
        c = np.concatenate([c_x[..., None], c_y[..., None]], axis=1)
        dist = np.linalg.norm(c, axis=1)
        dists[k] = np.mean(dist)
    return dists


def tracklet2bbox(track, num_frame):
    # assign_prev
    bbox = np.zeros((num_frame, 5))
    trackd = {}
    for k, v in track:
        bbox[k] = v
        trackd[k] = v
    for i in range(num_frame):
        if bbox[i][-1] <= 0.5:
            mind = np.Inf
            for k in trackd:
                if np.abs(k - i) < mind:
                    mind = np.abs(k - i)
            bbox[i] = bbox[k]
    return bbox

def tracklets2bbox(tracklet, num_frame):
    dists = distance_tracklet(tracklet)
    sorted_inds = sorted(dists, key=lambda x: dists[x])
    dist_thre = np.Inf
    for i in sorted_inds:
        if len(tracklet[i]) >= num_frame / 2:
            dist_thre = 2 * dists[i]
            break

    dist_thre = max(50, dist_thre)

    bbox = np.zeros((num_frame, 5))
    bboxd = {}
    for idx in sorted_inds:
        if dists[idx] < dist_thre:
            for k, v in tracklet[idx]:
                if bbox[k][-1] < 0.01:
                    bbox[k] = v
                    bboxd[k] = v
    bad = 0
    for idx in range(num_frame):
        if bbox[idx][-1] < 0.01:
            bad += 1
            mind = np.Inf
            mink = None
            for k in bboxd:
                if np.abs(k - idx) < mind:
                    mind = np.abs(k - idx)
                    mink = k
            bbox[idx] = bboxd[mink]
    return bad, bbox

def bboxes2bbox(bbox, num_frame):
    ret = np.zeros((num_frame, 2, 5))
    for t, item in enumerate(bbox):
        if item.shape[0] <= 2:
            ret[t, :item.shape[0]] = item
        else:
            inds = sorted(
                list(range(item.shape[0])), key=lambda x: -item[x, -1])
            ret[t] = item[inds[:2]]
    for t in range(num_frame):
        if ret[t, 0, -1] <= 0.01:
            ret[t] = ret[t - 1]
        elif ret[t, 1, -1] <= 0.01:
            if t:
                if ret[t - 1, 0, -1] > 0.01 and ret[t - 1, 1, -1] > 0.01:
                    if iou(ret[t, 0], ret[t - 1, 0]) > iou(
                            ret[t, 0], ret[t - 1, 1]):
                        ret[t, 1] = ret[t - 1, 1]
                    else:
                        ret[t, 1] = ret[t - 1, 0]
    return ret

def ntu_det_postproc(vid, det_results):
    det_results = [removedup(x) for x in det_results]
    label = int(vid.split('/')[-1].split('A')[1][:3])
    mpaction = list(range(50, 61)) + list(range(106, 121))
    n_person = 2 if label in mpaction else 1
    is_easy, bboxes = is_easy_example(det_results, n_person)
    if is_easy:
        print('\nEasy Example')
        return bboxes

    tracklets = bbox2tracklet(det_results)
    tracklets = drop_tracklet(tracklets)

    print(f'\nHard {n_person}-person Example, found {len(tracklets)} tracklet')
    if n_person == 1:
        if len(tracklets) == 1:
            tracklet = list(tracklets.values())[0]
            det_results = tracklet2bbox(tracklet, len(det_results))
            return np.stack(det_results)
        else:
            bad, det_results = tracklets2bbox(tracklets, len(det_results))
            return det_results
    # n_person is 2
    if len(tracklets) <= 2:
        tracklets = list(tracklets.values())
        bboxes = []
        for tracklet in tracklets:
            bboxes.append(tracklet2bbox(tracklet, len(det_results))[:, None])
        bbox = np.concatenate(bboxes, axis=1)
        return bbox
    else:
        return bboxes2bbox(det_results, len(det_results))

def pose_inference(args, frame_paths, det_results):
    
    print('Performing Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))

    num_frame = len(det_results)

    persons = [len(x) for x in det_results]
    num_person = max(persons) if len(persons) else 0
    kp = np.zeros((num_person, num_frame, 17, 3), dtype=np.float32)

    for i, (f, d) in enumerate(zip(frame_paths, det_results)):
        # Align input format
        d = [dict(bbox=x) for x in list(d) if x[-1] > 0.5]
        pose = inference_top_down_pose_model(posemodel, f, d, format='xyxy')[0]
        for j, item in enumerate(pose):
            kp[j, i] = item['keypoints']
        prog_bar.update()
    return kp

def ntu_pose_extraction(vid, skip_postproc=False):
    frame_paths = extract_frame(vid)
    det_results = detection_inference(args, frame_paths)
    if not skip_postproc:
        det_results = ntu_det_postproc(vid, det_results)
    pose_results = pose_inference(args, frame_paths, det_results)

    img = cv2.imread(frame_paths[0])
    h,w,_ = img.shape
    
    anno = dict()
    anno['keypoint'] = pose_results[..., :2]
    anno['keypoint_score'] = pose_results[..., 2]
    anno['frame_dir'] = osp.splitext(osp.basename(vid))[0]
    anno['img_shape'] = (h, w)
    anno['original_shape'] = (h, w)
    anno['total_frames'] = pose_results.shape[1]
    anno['label'] = int(osp.basename(vid).split('A')[1][:3]) - 1
    shutil.rmtree(osp.dirname(frame_paths[0]))

    return anno

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Pose Annotation for a single NTURGB-D video')
    parser.add_argument('video', type=str, help='source video')
    parser.add_argument('output', type=str, help='output pickle name')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--skip-postproc', action='store_true')
    args = parser.parse_args()
    return args

 
if __name__ == '__main__':
    root_dir = '/home/sckim/sckim/3.project/9.Action_Recognition/quantom/data/tmp'
    pkl_root_dir = '/home/sckim/sckim/3.project/9.Action_Recognition/quantom/data/pkl/custom_pkl'

    #Get Target Video Path List
    vid_list = os.listdir(root_dir)
    len_vid_list = len(vid_list)

    #Load Video and Get PKL File consists of NTU Dataset Format
    for index, vid_name in tqdm(enumerate(vid_list)):
        vid,ext = os.path.splitext(vid_name)
        args.video = os.path.join(root_dir,vid_name)
        args.output = os.path.join(pkl_root_dir,vid+'.pkl')

        #Get NTU Dataset Format Annotation
        anno = ntu_pose_extraction(args.video,args.skip_postproc)

        #Generate PKL File
        mmcv.dump(anno,args.output)


    #Merge and Divide to Train / Validation / Test Set
    output_train_pkl = '/home/sckim/sckim/3.project/9.Action_Recognition/quantom/data/pkl/custom_train.pkl' #0.85
    output_val_pkl = '/home/sckim/sckim/3.project/9.Action_Recognition/quantom/data/pkl/custom_val.pkl'   #0.1
    output_test_pkl = '/home/sckim/sckim/3.project/9.Action_Recognition/quantom/data/pkl/custom_test.pkl'  #0.05

    pkl_list = os.listdir(pkl_root_dir)
    len_pkl_list = len(pkl_list)
    data_train = []
    data_val = []
    data_test = []

    for index, pkl_name in tqdm(enumerate(pkl_list)):
        pkl_path = os.path.join(pkl_root_dir,pkl_name)
              
        if index<=0.85*len_pkl_list:
            with open(pkl_path,"rb") as pkl:
                data_train.append(pickle.load(pkl))        
        elif index<=0.95*len_pkl_list:
            with open(pkl_path,"rb") as pkl:
                data_val.append(pickle.load(pkl))        
        else:
            with open(pkl_path,"rb") as pkl:
                data_test.append(pickle.load(pkl))
        
        pkl.close()        
    
    #Generate Train / Vaidation / Test PKL File
    with open(output_train_pkl,"wb") as f_train:
                pickle.dump(data_train,f_train,protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_val_pkl,"wb") as f_val:
                pickle.dump(data_val,f_val,protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_test_pkl,"wb") as f_test:
                pickle.dump(data_test,f_test,protocol=pickle.HIGHEST_PROTOCOL)

    f_train.close()
    f_val.close()
    f_test.close()    