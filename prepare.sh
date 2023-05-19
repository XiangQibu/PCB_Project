#!/bin/bash
echo "start download weights!"

mkdir pretrained_weights
wget -O v5lite-s.pt https://cloud.tsinghua.edu.cn/f/adde50b8fe264de9a063/?dl=1
wget -O yolov5l.pt https://cloud.tsinghua.edu.cn/f/5491819f581844b2882e/?dl=1
wget -O yolov3.weights https://cloud.tsinghua.edu.cn/f/0a446f3c896e4e0a905e/?dl=1
mv v5lite-s.pt yolov5l.pt yolov3.weights ./pretrained_weights/

mkdir test_weights
wget -O YOLOv5_best.pt https://cloud.tsinghua.edu.cn/f/2a2af155cbbe4988957a/?dl=1
mv YOLOv5_best.pt test_weights

echo "download done!"

