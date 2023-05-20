import os
import random
import json
import os
from PIL import Image

# os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表

from os import listdir, getcwd
classes = ['drill_protru_in', 'drill_protru_out', 'drill_nick_in', 'drill_nick_out', 'drill_break', 'drill_open', 
           'open','nick', 'short','pinhole','island','protrusion','pad','copper','offset']
jsonfilepath = './data/Annotations'
txtsavepath = './data/ImageSets'
trainval_percent = 0.9
train_percent = 0.7
val_percent = 0.2
sets = ['train', 'test', 'val']
cls_id_list = []

# 进行归一化操作
def convert(w,h, box):  # size:(原图w,原图h) , box:(xmin,xmax,ymin,ymax)
    dw = 1./w     # 1/w
    dh = 1./h     # 1/h
    x = (box[0] + box[1])/2.0   # 物体在图中的中心点x坐标
    y = (box[2] + box[3])/2.0   # 物体在图中的中心点y坐标
    w = box[1] - box[0]         # 物体实际像素宽度
    h = box[3] - box[2]         # 物体实际像素高度
    x = x*dw    # 物体中心点x的坐标比(相当于 x/原图w)
    w = w*dw    # 物体宽度的宽度比(相当于 w/原图w)
    y = y*dh    # 物体中心点y的坐标比(相当于 y/原图h)
    h = h*dh    # 物体宽度的宽度比(相当于 h/原图h)
    return (x, y, w, h)    # 返回 相对于原图的物体中心点的x坐标比,y坐标比,宽度比,高度比,取值范围[0-1]


def convert_annotation(image_id,classes_processed):
    # 读取 JSON 文件
    # json解析
    with open('./data/Annotations/%s.json' % image_id, 'r') as f:
        annotation_list = json.load(f)
    img = Image.open('./data/images/%s.jpg' % image_id)
    w,h = img.size
    out_file = open('./data/labels/%s.txt' % (image_id), 'w', encoding='utf-8')
    # json: [{'label': 'nick', 'points': [[506, 446], [521, 484]]}]
    # 同一个PCB板中可能存在多个缺陷部位
    
    for defect_dic in annotation_list:
        defect_class = defect_dic.get('label')
        if defect_class not in classes_processed:
                continue
            # 通过类别名称找到id
        cls_id = classes_processed.index(defect_class)
        cls_id_list.append(cls_id)
        # box = (xmin,xmax,ymin,ymax)
        defect_box = (defect_dic.get('points')[0][0],
                      defect_dic.get('points')[1][0],
                      defect_dic.get('points')[0][1],
                      defect_dic.get('points')[1][1])
        bb = convert(w, h, defect_box)
            # bb 对应的是归一化后的(x,y,w,h)
            # 生成 calss x y w h 在label文件中
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

# 去除部分乱码、不存在、以及offset的标签
def json_filter(jsonfilepath):
    total_json = os.listdir(jsonfilepath)
    print('total_json_num:', len(total_json))
    classes_dic = {}
    json_list_processed = []
    for json_name in total_json:
        with open('./data/Annotations/%s' % json_name, 'r') as f:
            annotation_list = json.load(f)
        flag = 1
        for annotation in annotation_list:
            if annotation.get("label") == 'offset':
                flag = 0
            elif annotation.get("label") in classes:
                if annotation.get("label") not in classes_dic:
                    classes_dic[annotation.get("label")] = 1
                else:
                    classes_dic[annotation.get("label")] += 1
            else:
                flag = 0
        if flag:
            json_list_processed.append(json_name)
    return json_list_processed, classes_dic

# 划分训练集、验证集和测试集
def figure_id_write(json_list):
    num = len(json_list)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(num * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)
    if not os.path.exists('./data/ImageSets'):
        os.makedirs('./data/ImageSets')

    ftrainval = open('./data/ImageSets/trainval.txt', 'w')
    ftest = open('./data/ImageSets/test.txt', 'w')
    ftrain = open('./data/ImageSets/train.txt', 'w')
    fval = open('./data/ImageSets/val.txt', 'w')

    for i in list:
        name = json_list[i][:-5] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()

# 写入图片路径和标签
def label_write(classes_processed):
    wd = getcwd()
    print(wd)
    for image_set in sets:
        '''
        对所有的文件数据集进行遍历
        做了两个工作：
            1.将所有图片文件都遍历一遍，并且将其所有的全路径都写在对应的txt文件中去，方便定位
            2.同时对所有的图片文件进行解析和转化，将其对应的bundingbox 以及类别的信息全部解析写到label 文件中去
            3.最后再通过直接读取文件，就能找到对应的label 信息
        '''
        # 先找labels文件夹如果不存在则创建
        if not os.path.exists('./data/labels'):
            os.makedirs('./data/labels')
        # 读取在ImageSets/Main 中的train、test..等文件的内容
        # 包含对应的文件名称
        image_ids = open('./data/ImageSets/%s.txt' % (image_set)).read().strip().split()
        # 打开对应的2012_train.txt 文件对其进行写入准备
        list_file = open('./data/%s.txt' % (image_set), 'w')
        # 将对应的文件_id以及全路径写进去并换行
        for image_id in image_ids:
            list_file.write('%s/data/images/%s.jpg\n' % (wd, image_id))
            # 调用  year = 年份  image_id = 对应的文件名_id
            convert_annotation(image_id,classes_processed)
        # 关闭文件
        list_file.close()

if __name__ == '__main__':
    json_list_processed, classes_dic = json_filter(jsonfilepath)
    # 去除label中存在乱码的样本
    print(classes_dic)
    print(sum(classes_dic.values()))
    print([*classes_dic])
    print('processed_json_num:', len(json_list_processed))
    classes_processed = [*classes_dic]
    classes_processed.sort()
    print(classes_processed)

    figure_id_write(json_list_processed)
    label_write(classes_processed)

    
    # 返回当前工作目录
    print(set(cls_id_list))
