import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import warnings
import random
import torch.nn.functional as F

random.seed(3)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = []
        for path in self.img_files:
            image_dir = os.path.dirname(path)
            label_dir = "labels".join(image_dir.rsplit("images", 1))
            assert label_dir != image_dir, \
                f"Image path must contain a folder named 'images'! \n'{image_dir}'"
            label_file = os.path.join(label_dir, os.path.basename(path))
            label_file = os.path.splitext(label_file)[0] + '.txt'
            self.label_files.append(label_file)

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        try:

            img_path = self.img_files[index % len(self.img_files)].rstrip()

            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)
        except Exception:
            print(f"Could not read label '{label_path}'.")
            return

        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except Exception:
                print("Could not apply transform.")
                return

        return img_path, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)
""" class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        '''

        :param list_path: 一个txt文件，比如我们前面写的train_path.txt和val_path.txt
        :param img_size: 数据图片要转成成的高
        :param augment: 是否使用数据增强
        :param multiscale: 是否进行多尺度变换（看self.collate_fn就能明白它的作用）
        :param normalized_labels: 标签是否已经归一化，即boundingbox的中心坐标，高宽等是否已经归一化
        '''
        with open(list_path, "r") as file:
            self.img_files = file.readlines()   # 读取txt文件的内容，将样本路径读取出来

        # 标签的路径，可以根据样本的路径来获得，只需要将路径名中的images改成labels，后缀改成txt就行
        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size    # 图片处理成方形后的高宽（图片在输入模型前要处理成方形）
        self.max_objects = 100      # 一张图片中的最大目标数
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32  # 在进行多尺度变换时的最小尺度
        self.max_size = self.img_size + 3 * 32  # 在进行多尺度变换时的最大尺度
        self.batch_count = 0                    # 统计已经遍历了多少个batch
        # TODO max_objects是用来干嘛的
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()     # 获取图片的路径名

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))    # 读取图片并转化为torch张量
        # mage.open(img_path)读取图片，返回Image对象，不是普通的数组
        # convert('RGB')进行通道转换，因为当图像格式为RGBA时，Image.open(‘xxx.jpg’)读取的格式为RGBA

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))
        # 图片有可能是一张灰度图，那么img.shape就是（h, w）
        # unsqueeze(0)之后，就是img.shape就是（1, h, w）
        # img.expand((3, img.shape[1:])) 即为 img.expand((3, h, w))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # h_factor, w_factor在后面用来反算目标在图片中的具体坐标，看到本函数的后面，自然能明白
        # 如果已经归一化，那么比例因子就是图片的真实高宽
        # 如果未归一化，那比例因子就是1

        # Pad to square resolution
        img, pad = pad_to_square(img, 0)        
        _, padded_h, padded_w = img.shape
        label_path = self.label_files[index % len(self.img_files)].rstrip() # 获取标签路径

        targets = None
        if os.path.exists(label_path):
            f = open(label_path, 'r')
            if f.readlines() != []:
                # 有些图片没有目标，但有标签文件，这些标签文件中没有内容
                # 我们这边只处理有内容的标签文件，对于没有内容的标签文件，让targets等于None

                boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))

                # Extract coordinates for unpadded + unscaled image
                # 获取bbox左上角和右下角点在原始图片上的真实坐标
                x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
                y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
                x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
                y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)

                # Adjust for added padding
                # 由于已经被调整成了方形，因此需要加上pad的尺寸
                x1 += pad[0]
                y1 += pad[2]
                x2 += pad[1]
                y2 += pad[3]

                # Returns (x, y, w, h)
                # 求归一化后的中心点坐标和高宽
                boxes[:, 1] = ((x1 + x2) / 2) / padded_w
                boxes[:, 2] = ((y1 + y2) / 2) / padded_h
                boxes[:, 3] *= w_factor / padded_w
                boxes[:, 4] *= h_factor / padded_h

                targets = torch.zeros((len(boxes), 6))
                
                targets[:, 1:] = boxes          # 后面5列分别是bbox的位置和高宽，然后是分类索引
                # target第0列，根据后面的collate_fn函数，可以看到第0列是图片在batch中的索引
                

                # Apply augmentations
                # 随机进行水平翻转
                if self.augment:
                    if np.random.random() < 0.5:
                        img, targets = horisontal_flip(img, targets)
                
            f.close()

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))    # zip括号中的参数*开头，表示解压缩
        # 上条命令执行之后，paths, imgs, targets都将成为元组，
        # 以paths为例，上述命令执行后，paths将成为由两个图片路径构成的元组

        # Add sample index to targets 将图片在batch中的索引，加到target的第0个列
        for i, boxes in enumerate(targets):
            if boxes is not None:
                boxes[:, 0] = i     # i表示当前batch中的第i张图片

        # Remove empty placeholder targets 有些图片没有目标，那么它对应的标签就是None
        targets = [boxes for boxes in targets if boxes is not None]  # 保留非None的标签

        targets = torch.cat(targets, 0)     # 标签级联，targets在转化前是一个元组
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:  # 每10个batch，随机改变一下尺度
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
            # range函数的第三个参数是32，能保证随机获得的新尺寸是32的倍数，
            # 因为是backbone是32倍下采样，如果不是32的倍数，那么卷积核不能完全把图片扫描

        # 将图片缩放到指定尺寸
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])    #
        self.batch_count += 1

        # 图片缩放之后，之所以标签不用改变，是因为标签已经归一化了，所以无需转换

        return paths, imgs, targets """




def pad_to_square(img, pad_value):
    """
    该函数是将图片扩充成正方形
    :param img: 图片张量
    :param pad_value:   用来填充的值，即左右或者上下的条
    :return:
    """
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    
    # Determine padding 
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # (0, 0, pad1, pad2)和(pad1, pad2, 0, 0)，括号中的四个值，分别表示左右上下
    # 如果h小于w，那么就是在上下填充，否则在左右填充
    # 因为后面使用F.pad函数，第二个参数是pad，它是一个包含四个数的元组
    
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad




def resize(image, size):
    """
    将图片缩放成指定尺寸
    :param image: 图片张量
    :param size:  指定尺寸，高和宽都是这个值，也就是说，本函数缩放的是正方形
    :return:
    """
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image
