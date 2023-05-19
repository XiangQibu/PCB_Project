import numpy as np
import torch
import torch.nn as nn
from itertools import chain
from typing import List, Tuple
from utils.parse_config import parse_model_config
import torch.nn.functional as F
from utils.utils import to_cpu,weights_init_normal

def create_modules(blocks):
    """
    将blocks中的模块，都转换为模型，存入到nn.ModuleList()对象中
    其中shortcut和route模块，使用的都是EmptyLayer类，yolo模块使用的是DetectionLayer类

    :param blocks: cfg文件被解析后，每个模块的属性和值，都以键值对的形式存储在一个字典中，
            然后这些字典又被装入一个列表中，blocks就是这个列表
    :return:返回模型的超参数信息，和nn.ModuleList()对象（因为模型的输入尺寸、学习率、
            batch_size等，也都被存储在了cfg文件中）
    """
    net_info = blocks.pop(0)        # Captures the information about the input and pre-processing，即[net]层
    net_info.update({
        'batch': int(net_info['batch']),
        'subdivisions': int(net_info['subdivisions']),
        'width': int(net_info['width']),
        'height': int(net_info['height']),
        'channels': int(net_info['channels']),
        'optimizer': net_info.get('optimizer'),
        'momentum': float(net_info['momentum']),
        'decay': float(net_info['decay']),
        'learning_rate': float(net_info['learning_rate']),
        'burn_in': int(net_info['burn_in']),
        'max_batches': int(net_info['max_batches']),
        'policy': net_info['policy'],
        'lr_steps': list(zip(map(int,   net_info["steps"].split(",")),
                             map(float, net_info["scales"].split(","))))
    })
    assert net_info["height"] == net_info["width"], \
        "Height and width should be equal! Non square images are padded with zeros."
    
    
    module_list = nn.ModuleList()   #
    prev_filters = 3                # 初始通道数，因为彩色图片是RGB三通道，所以这里是3
    output_filters = [prev_filters]             # 每一层的输出通道，方便路由层（route层）追踪

    for index, x in enumerate(blocks):  # 之所以从第1块开始，是因为第0块是[net]层
        module = nn.Sequential()
        # 之所以要在这里建立一个nn.Sequential对象，是因为一个模块可能有卷积层、BN层、激活层等，
        # 所以需要先统一装到一个容器中，这样才能装入到模型列表里面

        # check the type of block
        # create a new module for the block
        # append to module_list

        if (x["type"] == "convolutional"):  # 如果是卷积模块
            # Get the info about the layer
            bn = int(x["batch_normalize"])
            filters = int(x["filters"])
            kernel_size = int(x["size"])
            pad = (kernel_size - 1) // 2
            module.add_module(
                f"conv_{index}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(x["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                module.add_module(f"batch_norm_{index}",
                                   nn.BatchNorm2d(filters, momentum=0.1, eps=1e-5))

            """ # 获取卷积层（Conv2d）的相关参数
            try:
                batch_normalize = int(x["batch_normalize"])
                # 为了防止卷积模块没有BN层，所以加入到try当中
                bias = False    # 只要有BN，那么就相当于没有偏置，因为即便有，也会被BN给抹平
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])     # backbone中只要是卷积模块，就都会设置pad
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                # 如果设置 pad=0，那么padding就是0，说明没有填充
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)       # 将卷积层加入到容器中

            # Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)   # 将BN层加入到容器中
 """

            # Check the activation.
            # It is either Linear or a Leaky ReLU for YOLO
            activation = x["activation"]
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)    # 将激活层加入到容器中

        # If it's an upsampling layer
        # We use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="bilinear")     # TODO 这里会出现警告
            module.add_module("upsample_{}".format(index), upsample)

        # If it is a route layer
        elif (x["type"] == "route"):
            

            # Start of a route
            start = int(x["layers"].split(',')[0])     # 路由层可能从前面第4层牵出一条线
            # end, if there exists one.如果只有一条线，那么到这里就结束了

            # 因为也有可能存在第二条线，所以这里使用异常机制
            try:
                end = int(x["layers"].split(',')[1])
            except:
                end = 0

            # Positive annotation，如果route定位用的数字是正的，比如61，那么说明是第61层
            if start > 0:
                start = start - index   # 第一条线的层数，减当前层数，得到的start为第一条线相对于本层的相对位置
            if end > 0:
                end = end - index       # 第二条线的层数，减当前层数，得到的start为第二条线相对于本层的相对位置

            # route = EmptyLayer()        # 一个空层对象，在其他文件中定义
            module.add_module("route_{0}".format(index), nn.Sequential())

            # 获得route层的输出通道
            if end < 0:
                # 说明存在第二条线，因为当存在第二条线时，
                # 如果为绝对层数（位置），那么必然已经执行了end = end - index的操作，end必然已变为为负
                # 如果为相对层数（位置），那么end本身就是负的
                filters = output_filters[index + start + 1] + output_filters[index + end + 1]
            else:
                # 如果不存在第二条线
                # 如果不是负的，那么必然为0，因为不可能是正的，end为0说明不存在第二条线
                filters = output_filters[index + start + 1]

        # shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            # shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), nn.Sequential())

        # Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
            # 比如yolo1，mask = 0,1,2，经过上面的命令之后，变为[0, 1, 2]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            # 最初anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
            # yolo1的mask为[0, 1, 2]，经过上面的操作
            # yolo1的anchors为[[10,13], [16,30], [30,61]]
            num_classes = int(x["classes"])
            new_coords = bool(x.get("new_coords", False))

            yolo = YOLOLayer(anchors, num_classes,new_coords)
            module.add_module("Detection_{}".format(index), yolo)

        # 将上面解析得到的模块加入到module_list（即循环之前定义的nn.ModuleList对象）中
        module_list.append(module)

        # 上面的各个模块，只有拼接模块和卷积模块会改变通道，
        # 所以只在 route 和 convolutional 中有给 filters 赋值
        # 其他模块仍然使用上一轮循环得到的通道
        output_filters.append(filters)          # 将当前模块的输出通道数存入output_filters中

        # 将当前模块的输出通道赋值给 prev_filters，用于下一轮循环
        prev_filters = filters

    return (net_info, module_list)

# 空层,先用来占位
class EmptyLayer(nn.Module):    # 这个类是用来给route和shortcut模块凑数的，解析cfg文件时用到
    def __init__(self):
        super(EmptyLayer, self).__init__()
    # 为何没有forward函数？请仔细看Darknet类forward方法中的module_type == "route"的实现代码
    # 相当于将EmptyLayer的forward放到了Darknet类的forward中
    # 无论是route模块还是shortcut模块，都需要获得前面层的输出，这个在这里不太好实现
    # 所以这里写了一个加模块凑数，因为nn.ModuleList中的元素必须为nn.Module的子类

# 上采样层
class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        # 通过插值实现上采样
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

# YOLO层
class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors: List[Tuple[int, int]], num_classes: int, new_coords: bool):
        """
        Create a YOLO layer

        :param anchors: List of anchors
        :param num_classes: Number of classes
        :param new_coords: Whether to use the new coordinate format from YOLO V7
        """
        super(YOLOLayer, self).__init__()
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.new_coords = new_coords
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.no = num_classes + 5  # number of outputs per anchor
        self.grid = torch.zeros(1)  # TODO

        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2)
        self.register_buffer('anchors', anchors)
        self.register_buffer(
            'anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))
        self.stride = None

    def forward(self, x: torch.Tensor, img_size: int) -> torch.Tensor:
        """
        Forward pass of the YOLO layer

        :param x: Input tensor
        :param img_size: Size of the input image
        """
        # print(x.shape)
        stride = img_size // x.size(2)
        self.stride = stride
        bs, _, ny, nx = x.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x = x.view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        if not self.training:  # inference
            if self.grid.shape[2:4] != x.shape[2:4]:
                self.grid = self._make_grid(nx, ny).to(x.device)

            if self.new_coords:
                x[..., 0:2] = (x[..., 0:2] + self.grid) * stride  # xy
                x[..., 2:4] = x[..., 2:4] ** 2 * (4 * self.anchor_grid) # wh
            else:
                x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid) * stride  # xy
                x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor_grid # wh
                x[..., 4:] = x[..., 4:].sigmoid() # conf, cls
            x = x.view(bs, -1, self.no)

        return x

    @staticmethod
    def _make_grid(nx: int = 20, ny: int = 20) -> torch.Tensor:
        """
        Create a grid of (x, y) coordinates

        :param nx: Number of x coordinates
        :param ny: Number of y coordinates
        """
        # yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()




class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)     # 解析配置文件，返回由block构成的列表
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        # create_modules方法是将每个block转化为nn.Sequential()对象，然后再加入到nn.ModuleList()对象中
        
        self.yolo_layers = [layer[0]
                            for layer in self.module_list if isinstance(layer[0], YOLOLayer)]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)
    def forward(self, x):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs = []      # 记录每一个模块的输出，方面route和shortcut模块的追踪
        yolo_outputs = []       # 记录每一个yolo模块的输出，方便将不同yolo模块的输出进行级联
        # print(self.module_defs)
        # print(zip(self.module_defs, self.module_list))
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            # 这里module_def是一个解析cfg文件后的块，module是一个module_list对象
            # print('layer:',i,'module_type:',module_def["type"])
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                # print(module_def['layers'])
                output = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
                group_size = output.shape[1] // int(module_def.get("groups", 1))
                group_id = int(module_def.get("group_id", 0))
                x = output[:, group_size * group_id : group_size * (group_id + 1)] # Slice groupings used by yolo v4
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x = module[0](x, img_dim)  # 计算输出
                # print('yolo layer x.shape:',x.shape)
                yolo_outputs.append(x)
            # print('x.shape: ',x.shape)
            layer_outputs.append(x)
        # 将几个yolo_outputs进行级联，yolo_ouput是列表
        return yolo_outputs if self.training else torch.cat(yolo_outputs, 1)
    
    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:         # 以二进制方式打开文件
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights 将二进制文件转化为numpy数组

        # Establish cutoff for loading backbone weights
        # 如果只想导入backbone，那么就需要设定一个cutoff变量，用于停止导入数据
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0     # 这就类似于一个指针，确定每次导入数据从哪个位置开始
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                # 当i为75时，说明backbone部分的迁移学习结束了，此时退出循环
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # print(i)
                # print(i)
                # print(conv_layer)
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

def load_model(model_path, weights_path=None):
    """Loads the yolo model from file.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :return: Returns model
    :rtype: Darknet
    """
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")  # Select device for inference
    model = Darknet(model_path).to(device)

    model.apply(weights_init_normal)

    # If pretrained weights are specified, start from checkpoint or weight file
    if weights_path:
        if weights_path.endswith(".pth"):
            # Load checkpoint weights
            model.load_state_dict(torch.load(weights_path, map_location=device))
        else:
            # Load darknet weights
            model.load_darknet_weights(weights_path)
    return model

if __name__ == '__main__':
    """测试两种迁移学习"""

    test_input = torch.rand((2, 3, 416, 416))

    # 建立模型并导入配置文件
    file_path = r"./config/yolov3.cfg"
    model = Darknet(file_path)
    pred0 = model(test_input)
    #print(np.array(pred0).shape)
    print(len(pred0[0][0]))

    # 第一种：导入整个模型的参数文件
    model.load_darknet_weights(r"./weights/yolov3.weights")
    pred1 = model(test_input)
    #print(pred1.shape)
    #print(pred1)

	# 第二种：导入backbone的参数文件
    model.load_darknet_weights(r"weights/darknet53.conv.74")
    pred2 = model(test_input)
    print('-'*50)
    #print(pred2.shape)
    #print(pred2)

