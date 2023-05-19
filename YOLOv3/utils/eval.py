import tqdm
from torch.autograd import Variable
import numpy as np
import time
from utils.datasets import ListDataset
from utils.utils import *


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    """

    :param model:Darknet模型对象
    :param path: 存放了验证集文件名的文件
    :param iou_thres: iou阈值，
    :param conf_thres:置信度阈值
    :param nms_thres:
    :param img_size:
    :param batch_size:
    :return:
    """
    # 把模型切换到评估模式
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

        # Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    Tensor = torch.FloatTensor      # TODO 本来应该根据model的device来选择设备，这里我们简单点

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):  # 使用tqdm打印进度条，便于观察进度
        # tqdm.tqdm(iter, desc) iter是可迭代对象，desc是描述，用于显示迭代进度

        # Extract labels
        labels += targets[:, 1].tolist()    # 抽取标签中的类别信息

        # Rescale target 转换标签值，以便后面可以和预测值进行比较
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])  # boundingbox的数据原本是中心坐标和高宽，现在换成了左上角和右下角角点坐标
        targets[:, 2:] *= img_size                  # 因为boundingbox是目标值，所以将其转化为图片中的真实坐标

        imgs = imgs.to(device)
        # print('\nimgs device type: ',imgs.device.type)
        # print('\nmodel device type: ', next(model.parameters()).is_cuda)


        with torch.no_grad():
            
            outputs = model(imgs)                   # 使用模型进行预测
            
            # 
            t_input = time.time()
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)  # 非极大值抑制
            t_output = time.time()
            print("non_max_suppression time:", t_output - t_input)
            # outputs是一个列表，具体看non_max_suppression方法

        # 获得当前batch中，每一张图片经过NMS之后的结果
        t_input = time.time()
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
        t_output = time.time()
        print("get_batch_statistics time:", t_output - t_input)
    # Concatenate sample statistics
    L = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]  # x每次获得的都是一个元组，
    # sample_metrics是一个大列表，其中每个元素都是一个小列表
    # 每次获得的x都是从小列表中抽取一个元素（该元素是张量），构成的一个元组
    # x是元组里面套了若干个张量（有多少个batch，就有多少个张量）
    # np.concatenate是将这几个张量进行级联，相当于原来目标是分散在各个batch中，现在组合在在一起
    # 最后的结果是L里面套了三个numpy数组，分别是true_positives，pred_scores和pred_labels
    # 这三个数组都是代表了整个验证集，而非仅仅单个batch

    print(len(L))
    true_positives, pred_scores, pred_labels = L[0], L[1], L[2]

    # true_positives, pred_scores, pred_labels都是numpy数组
    # 这里求统计指标时，就不再是一个batch一个batch地求了，而是在整个验证集上求
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class



def xywh2xyxy(x):
    """
    边框信息转化
    :param x: 由中心点坐标和高宽构成的边框
    :return: 由上下角点坐标构成的边框
    """
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    :param prediction: 维度为(batch_size, number_pred, 85)，number_pred是三个检测头的数量之和
    :param conf_thres: 置信度阈值
    :param nms_thres: 非极大值抑制时的iou阈值
    :return: 输出一个列表，列表中的元素，要么为None，要么是维度为(num_obj, 7)的张量
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])        # 将预测得到的boundingbox标签转化为两个个角点的坐标
    output = [None for _ in range(len(prediction))]             # 先获得由若干None构成的列表
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]     # 先筛选置信度大于阈值的预测框
        # If none are remaining => process next image
        if not image_pred.size(0):         # 如果当前图片中，所有目标的置信度都小于阈值，那么就进行下一轮循环
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # image_pred[:, 4]是置信度，image_pred[:, 5:].max(1)[0]是概率最大的类别索引

        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        # argsort()是将(-score)中的元素从小到大排序，返回排序后索引
        # 将(-score)中的元素从小到大排序，实际上是对score从大到小排序
        # 将排序后的索引放入image_pred中作为索引，实际上是对本张图片中预测出来的目标进行排序

        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)   #
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # 经过上条命令之后，detections的维度为(number_pred, 7)，
        # 前4列是边框的左上角点和右下角点的坐标，第5列是目标的置信度，第6列是类别置信度的最大值，
        # 第7列是类别置信度最大值所对应的类别

        # Perform non-maximum suppression
        keep_boxes = []     # 用来存储符合要求的目标框

        while detections.size(0):   # 如果detections中还有目标
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            # bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4])返回值的维度为(num_objects, )
            # bbox_iou的返回值与非极大值抑制的阈值相比较，获得布尔索引
            # 即剩下的边框中，只有detection[0]的iou大于nms_thres的，才抑制，即认为这些边框与detection[0]检测的是同一个目标

            label_match = detections[0, -1] == detections[:, -1]
            # 布尔索引，获得所有与detection[0]相同类别的对象的索引

            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match   # &是位运算符，两个布尔索引进行位运算
            # 上面位运算得到的索引，是所有应该被抑制的边框的索引，即无效索引
            # 所谓无效索引，即和最大置信度边框有相同类别，而且有比较高的交并比的边框的索引
            # 这里是筛选出无效边框，只有被筛选出来的边框才需要被抑制

            weights = detections[invalid, 4:5]      # 根据无效索引，获得被抑制边框的置信度

            # 加权获得最后的边框坐标
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            # 上面的命令是将当前边框，和被抑制的边框，进行加权，类似于好几个边框都检测到了同一张人脸，
            # 将这几个边框的左上角点横坐标x进行加权（按照置信度加权），获得最后边框的x
            # 对左上角点的纵坐标y，以及右下角点的横纵坐标也进行加权处理

            keep_boxes += [detections[0]]       # 将有效边框加入到 keep_boxes 中
            detections = detections[~invalid]   # 去掉无效边框，更新detections

        if keep_boxes:  # 如果keep_boxes不是空列表
            output[image_i] = torch.stack(keep_boxes)   # 将目标堆叠，然后加入到列表
            # 假设NMS之后，第i张图中有num_obj个目标，那么torch.stack(keep_boxes)的结果是就是一个(num_obj, 7)的张量，没有图片索引

        # 如果keep_boxes为空列表，那么output[image_i]则未被赋值，保留原来的值（原来的为None）

    return output



def get_batch_statistics(outputs, targets, iou_threshold):
    """
    outputs是模型的预测结果，但预测的边框未必都合格，必须从中筛选出合格的边框，所谓合格，
    就是预测框和真实框的iou大于阈值，这里仅仅考虑iou，不考虑预测类别是否相同
    :param outputs:一个列表，列表中第i个元素，要么是维度为(num_obj, 7)的张量（第i张图片的预测结果），
                    要么是None（即模型认为该张图片中没有目标）
    :param targets:一个张量，这个张量的首列是图片在batch中的索引，第2列是类别标签，
                后面4列分别是box的上下角点横纵坐标
    :param iou_threshold:预测的边框与GT的阈值，经过NMS之后，留下的边框必须与GT的iou超过这个阈值才能参与评价
    :return:[A, B, C，……]，  A、B、C都是列表，他们分别代表包含了合格预测框的图片
            列表A由三个元素组成，即[true_positives, pred_scores, pred_labels]，列表B，C也是如此
            true_positives 一维张量，代表当前图片的所有预测边框中，合格边框的布尔索引
            pred_scores 一维张量，代表当前图片的所有预测边框的置信度，
            pred_labels 一维张量，代表当前图片的所有预测边框的类别标签
    """
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):
        # sample_i是图片索引

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]      # 获取第i张图片的预测结果
        pred_boxes = output[:, :4]      # 从输出中获得预测框的上下角点的坐标
        pred_scores = output[:, 4]      # 获得置信度
        pred_labels = output[:, -1]     # 获得类别索引

        true_positives = np.zeros(pred_boxes.shape[0])  # TP，即真实正样本，详情要看后面的循环
        # pred_boxes.shape[0]是第i张图片中的目标数量

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        # 从目标张量中获得第i张图片的标签
        # 从之所以从1开始，是因为0是图片索引，这里是遍历图片，所以不需要图片索引

        target_labels = annotations[:, 0] if len(annotations) else []
        # 获取目标的类别索引，若图片中无目标则返回空列表
        # targets和output的排列方式不一样，targets中第一列是图片索引，第二列是类别索引

        if len(annotations):
            detected_boxes = []                 # 用来存储预测合格的边框的索引，
            # 所谓预测合格，是指与GT的iou大于阈值
            target_boxes = annotations[:, 1:]   # 从标签中获得边框的信息

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                # zip(pred_boxes, pred_labels)是每次分别从pred_boxes和pred_labels中抽取一个元素组成元组

                # If all targets are found break
                if len(detected_boxes) == len(annotations):
                    # 当条件成立时，说明标签中的目标边框已经被匹配完，可以跳出循环了
                    # 另外一种结束循环的方法是zip(pred_boxes, pred_labels)被遍历完
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    # 说明类别预测错误
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                # pred_box.unsqueeze(0)之后，pred_box的维度为(1, 4)，
                # target_boxes的维度为 (num_obj, 4)
                # bbox_iou(pred_box.unsqueeze(0), target_boxes)的返回值是pred_box与当前图片中所有GT的iou
                # 选择最大的iou及其在targets中的索引

                if iou >= iou_threshold and box_index not in detected_boxes:
                    # 如果最大的iou大于阈值，且这个边框对应的GT索引不在detected_boxes中，
                    # 那么就将这个边框对应的GT索引加入到detected_boxes中，并在TP中将对应的位置标1

                    detected_boxes += [box_index]
                    true_positives[pred_i] = 1

        # 经过上面的循环后true_positives的元素个数，表示的是模型预测有多少个目标，
        # 这些预测的目标如果合格（与targets的最大iou大于阈值），就在对应位置标1，否则保留原来的值（即为0）
        # true_positives可以作为pred_scores和pred_labels的布尔索引

        # 将当前图片的true_positives，pred_scores, pred_labels加入到batch_metrics中
        batch_metrics.append([true_positives, pred_scores, pred_labels])

    return batch_metrics

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Mricset.
    # Arguments
        tp:    一维数组
        conf:  一维数组
        pred_cls: 一维数组
        target_cls: 一维数组
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)   # 将conf从大到小排序，返回排序后的索引
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]    # 将tp, conf, pred_cls从大到小排序

    # Find unique classes
    unique_classes = np.unique(target_cls)     # 将当前batch中所有GT的类别信息进行去重

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"): # 这个循环是计算每个类别的AP值
        i = pred_cls == c               # 返回布尔值，即是否为当前类别
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()                   # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            # 如果当前类别的GT和预测框的数量同时为0，
            # 这种情况基本不可能发生，因为c的来源就是GT，至少GT不可能等于0
            continue
        elif n_p == 0 or n_gt == 0:     #
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            # tp本来就是合格边框的布尔索引，经过i又一轮筛选后，就成为了当前类别的合格边框的布尔索引
            # (1 - tp[i]).cumsum() 先进行广播，再计算轴向元素累加和

            tpc = (tp[i]).cumsum()          # 将当前类别的布尔索引进行累加 结果可能是0，1，1,4

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)     # 召回率曲线计算
            r.append(recall_curve[-1])              # recall_curve[-1]是不设置信度阈值时的召回率

            # Precision
            precision_curve = tpc / (tpc + fpc)     # 准确率曲线计算
            p.append(precision_curve[-1])           # precision_curve[-1]是不设置信度阈值时的准确率

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)       # 将准确率、召回率、AP做成numpy数组
    f1 = 2 * p * r / (p + r + 1e-16)                        # 求F1分数

    # 返回统计指标
    # 这里之所以要把unique_classes一起返回，因为p[i], r[i], ap[i]等统计指标，
    # 是和unique_classes[i]表示的类别对应的
    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
    # np.where(condition)表示找到符合条件的索引

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
