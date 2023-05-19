# 基于机器学习方法的自动光学检测

## 准备工作
提交的代码包主目录文件构成如下：
> --config
> --detect
> --YOLOv3
> --YOLOv5
> --YOLOv5-Lite
> --data_process.py
> --K_means.py
> --README.md
> --prepare.sh
> --requirements.txt

#### 运行环境
1. 系统版本：
- Distributor ID: Ubuntu
- Description:    Ubuntu 16.04.5 LTS
- Release:        16.04
- Codename:       xenial
- cuda版本: 10.2
2. Annaconda 版本
conda 4.10.3
<https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2021.11-Linux-x86_64.sh>

3. 虚拟环境创建
```
conda create --name PCB_env python=3.6.13
conda activate PCB_env
pip install -r requirements.txt
```

4. wandb准备(可选)
```
pip install wandb==0.15.0
```
本项目可以进行可视化监测，可以进行wandb准备.
  1. 进入<https://wandb.ai/>，登录/创建帐号，复制自己账号的API key。
  2. 在服务器终端输入```wandb login```，粘贴API key。

#### 数据集下载
从<https://cloud.tsinghua.edu.cn/d/7335cda05c364dffadcf/>下载PR_AIVS.zip到主目录下。运行以下指令。

```
unzip PR_AIVS.zip
mkdir ./data 
mkdir ./data/Annotations
mkdir ./data/images
mv ./PR_AIVS/*.jpg ./data/images/
mv ./PR_AIVS/*.json ./data/Annotations/
```

#### 预训练模型下载
预训练模型以及最优模型已经上传至清华云盘，这里只需要运行
```
bash prepare.sh
```
即可完成下载。

## 数据预处理
在报告中已经介绍过，在训练模型之前，需要对数据进行筛选和整理，运行
```
python data_process.py
```

## 模型训练
三种模型需要分别训练。
- YOLOv3

```
python ./YOLOv3/train.py
# 或者
python ./YOLOv3/train.py --epochs 200 --model_def ./config/yolov3.cfg --data_config ./config/PCB.yaml --pretrained_weights ./pretrained_weights/yolov3.weights
```
模型文件会存于./checkpoints 中

- YOLOv5-Lite
```
python ./YOLOv5-Lite/train.py
# 或者
python ./YOLOv5-Lite/train.py --epochs 100 --weights ./pretrained_weights/v5lite-s.pt --cfg ./config/v5Lite-s.yaml --data ./config/PCB.yaml
```
训练结果存于./runs/train中
- YOLOv5(**最优模型**)
```
python ./YOLOv5/train.py
```
训练结果存于./runs/train中

## 模型测试
使用最优模型对测试集数据进行测试
```
# 使用最优模型
python ./detect/val.py --weights ./test_weights/YOLOv5_best.pt --task test

# 使用上一步训练出的模型
# 注意：训练得到的权重.pt文件在./runs/train下的哪个文件夹需要视训练日志而定
python ./detect/val.py --weights ./runs/train/exp/weights/best.pt --task test

```

## 可视化缺陷检测
对输入的图像进行可视化缺陷检测
```
python ./detect/detect.py --weights ./test_weights/YOLOv5_best.pt --source ./data/test.txt --save-txt
```

## K_means.py
该脚本是实现基于数据集进行边框聚类，最终输出聚类信息。
聚类的配置已经在模型中修改了，所以本脚本对模型训练不造成影响。
直接运行即可
```
python K_means.py
```

## 代码参考说明
本项目中的代码，部分借鉴于开源代码，在此处列出参考链接。
- ./YOLOv5 & ./detect
  参考<https://github.com/ultralytics/yolov5>
- ./YOLOv5-Lite
  参考<https://github.com/ppogg/YOLOv5-Lite>
- ./YOLOv3/utils & ./YOLOv3/test.py
  参考<https://github.com/eriklindernoren/PyTorch-YOLOv3/tree/master/pytorchyolo>

