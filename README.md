# Action Recognition

It is revealed that this project is based on [Open MMProject](https://github.com/open-mmlab).
<br>This Project's Goal is to Apply Action Recognition with Multi GPU and TensorRT.</br>

## Configuration

The configuration is as follows  
1. preprocessing : we offer merge dataset(pkl file) function
2. visualization : we offer visualization of confidence score map which is input of posec3d
3. inference : we offer demo, inference on single gpu, inference on multi gpu with tensorRT and Flask
<br></br>
> we modify and custom mmaction2, mmpose, mmdetection project's some part about detection target to use the tensorRT and multi gpu.   

## Inference
```
python inference_trt_multi.py --det-config [detection model config path] --det-checkpoint [detection model checkpoint path] --det-deploy-config [detection model deploy config path] --pose-config [pose estimation model config path] --pose-checkpoint [pose estimation model checkpoint path --skeleton-config [action recognition model config path] --pose-deploy-config [pose estimation deploy config path] --skeleton-checkpoint [action recognition model checkpoint path] --video_folder [video src path] --det-batch [detection batch size] --pose-batch [pose estimation batch size] --action-batch [action recognition batch size]
```

## Result

1) Visualization of Confidence Map
<br></br>
![1128010132586184](https://user-images.githubusercontent.com/63839581/204144892-5137f335-6807-4e88-b15f-21a2b80acce9.jpg)
<br></br>
```
python visualization/get_confmap.py --det-config [detection model config path] --det-checkpoint [detection model checkpoint path] --pose-config [pose estimation model config path] --pose-checkpoint [pose estimation model checkpoint path] --video_folder [video src path] --video_out_folder [video dst path]
```

2) Visualization of Demo Video
<br></br>
![ezgif com-gif-maker](https://user-images.githubusercontent.com/63839581/204144338-acbe7ada-2e88-45ca-8f53-fb22a0105611.gif)
<br></br>
```
python demo.py --det-config [detection model config path] --det-checkpoint [detection model checkpoint path] --pose-config [pose estimation model config path] --pose-checkpoint [pose estimation model checkpoint path --skeleton-config [action recognition model config path] --skeleton-checkpoint [action recognition model checkpoint path] --video_folder [video src path] --video_out_folder [video dst path]
```
