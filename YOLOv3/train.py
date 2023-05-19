from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from utils.eval import evaluate
from utils.loss import compute_loss
from utils.augmentations import AUGMENTATION_TRANSFORMS

import warnings
warnings.filterwarnings("ignore")   # 过滤掉警告

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from test import _evaluate, _create_validation_data_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="./config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="./config/PCB.yaml", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, default="./pretrained_weights/yolov3.weights", help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=100, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--logdir", default='logs', help="Directory for training log files (e.g. for TensorBoard)")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="Evaluation: IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="Evaluation: Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="Evaluation: IOU threshold for non-maximum suppression")
    
    parser.add_argument("--device", default='0', help='Gpu name')
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the training more verbose")
    opt = parser.parse_args()
    print(opt)

    logger = Logger(opt.logdir)  # Tensorboard logger
    # 确定使用GPU还是CPU
    torch.cuda.set_device(int(opt.device))
    device = torch.device('cuda:%d' % int(opt.device) if torch.cuda.is_available() else "cpu")

    # print(torch.cuda.current_device())
    # 解析数据配置文件
    data_config = parse_data_config(opt.data_config)  # 解析关于数据的配置文件，即config/coco.data
    train_path = data_config["train"]             # 获取训练集样本路径地址
    valid_path = data_config["val"]               # 获取验证集样本路径地址
    class_names = data_config["names"]  # 获得类别名

    # 建立模型并初始化
    # model = Darknet(opt.model_def).to(device)  # 导入配置文件建立模型，并放入GPU中
    # model.apply(weights_init_normal)  # 权重初始化，weights_init_normal是utils/utils.py中的方
    # model.apply(weights_init_normal)表示对model中的每个参数都使用weights_init_normal方法进行初始化
    model = load_model(opt.model_def, opt.pretrained_weights)
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):  # 有可能导入的是整个模型
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:  # 也有可能导入的是模型的权重
            model.load_darknet_weights(opt.pretrained_weights)
            print('weights loaded!')

    # Get dataloader
    dataset = ListDataset(train_path, multiscale=opt.multiscale_training,transform=AUGMENTATION_TRANSFORMS)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,  # 是否在返回前，将数据复制到显存中
        collate_fn=dataset.collate_fn,
    )

    
    validation_dataloader = _create_validation_data_loader(
        valid_path,
        opt.batch_size,
        model.hyperparams['height'],
        opt.n_cpu)

    optimizer = torch.optim.Adam(model.parameters(),
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],)  # 指定优化器
    """ optimizer = optim.SGD(
            model.parameters(),
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
            momentum=model.hyperparams['momentum']) """

    #----------------------train---------------------------------------
    for epoch in range(opt.epochs):
        model.train()               # 切换到训练模式
        start_time = time.time()    # 记录时间
        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
            batches_done = len(dataloader) * epoch + batch_i    
            # batches_done表示第几次迭代

            # 图片和标签做成变量
            imgs = imgs.to(device)
            targets = targets.to(device)
            # print(targets.shape)

            outputs = model(imgs)	# 前向传播
            loss, loss_components = compute_loss(outputs, targets, model)
            loss.backward()                         # 根据损失函数更新梯度


            if batches_done % opt.gradient_accumulations:
                # 学习率调整
                lr = model.hyperparams['learning_rate']
                if batches_done < model.hyperparams['burn_in']:
                    # Burn in
                    lr *= (batches_done / model.hyperparams['burn_in'])
                else:
                    # Set and parse the learning rate to the steps defined in the cfg
                    for threshold, value in model.hyperparams['lr_steps']:
                        if batches_done > threshold:
                            lr *= value
                # Log the learning rate
                logger.scalar_summary("train/learning_rate", lr, batches_done)
                # Set learning rate
                for g in optimizer.param_groups:
                    g['lr'] = lr

                # Accumulates gradient before each step
                # 这里并非每次得到梯度就更新，而是累积若干次梯度才进行更新
                optimizer.step()
                optimizer.zero_grad()   # 梯度信息清零

            # ############
            # Log progress
            # ############
            if opt.verbose:
                print(AsciiTable(
                    [
                        ["Type", "Value"],
                        ["IoU loss", float(loss_components[0])],
                        ["Object loss", float(loss_components[1])],
                        ["Class loss", float(loss_components[2])],
                        ["Loss", float(loss_components[3])],
                        ["Batch loss", to_cpu(loss).item()],
                    ]).table)

            # Tensorboard logging
            tensorboard_log = [
                ("train/iou_loss", float(loss_components[0])),
                ("train/obj_loss", float(loss_components[1])),
                ("train/class_loss", float(loss_components[2])),
                ("train/loss", to_cpu(loss).item())]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)

            model.seen += imgs.size(0)
        
        if (epoch + 1) % opt.checkpoint_interval == 0:
            # 保存模型
            if not os.path.exists('./checkpoints'):
                os.makedirs('./checkpoints')
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)


        if epoch % opt.evaluation_interval == 0:
            # 并不是每个epoch结束后都进行评价，而是若干个epoch结束后做一次评价
            print("\n---- Evaluating Model ----")

            # Evaluate the model on the validation set 将模型放在验证集上进行评价
            metrics_output = _evaluate(
                model,
                validation_dataloader,
                class_names,
                img_size=model.hyperparams['height'],
                iou_thres=opt.iou_thres,
                conf_thres=opt.conf_thres,
                nms_thres=opt.nms_thres,
                verbose=opt.verbose
            )

            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                evaluation_metrics = [
                    ("validation/precision", precision.mean()),
                    ("validation/recall", recall.mean()),
                    ("validation/mAP", AP.mean()),
                    ("validation/f1", f1.mean())]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)
        
        
        


















