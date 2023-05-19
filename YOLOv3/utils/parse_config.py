# prase_config.py
import yaml
def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions
    该函数以配置文件作为输入，即解析配置文件
    解析时将每个块存储为字典，这些块的属性和值都以键值对的形式存储在字典中。
    解析过程中，我们将这些字典（由代码中的变量 block 表示）添加到列表 blocks 中。
    我们的函数将返回该 blocks。"""

    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]   # 去掉空行和注释
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})  # 往列表里添加一个空字典
            module_defs[-1]['type'] = line[1:-1].rstrip()   # module_defs[-1]是一个字典，
            # 上面这条语句的意思是去掉空字符，并加入到字典里，因为方括号开头的行是像这个样子的：[convolutional]

            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0      # TODO 为何这里BN要预先设为0？
        else:
            # 如果不是模块开始的标志，那么就直接去掉两侧的空字符，然后用等号将一行分割成键值对加入到字典中
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

def parse_data_config(path):
    """Parses the data configuration file，返回一个字典，该字典为数据配置文件的信息"""
    options = dict()
    options['gpus'] = '0,1,2,3'     # TODO 这里为何要加上GPU的信息
    options['num_workers'] = '10'   # TODO 这里为何要加上num_workers？
    with open(path) as f:
        data_conf = yaml.load(f, Loader=yaml.FullLoader)
    options['train'] = data_conf['train']
    options['val'] = data_conf['val']
    options['test'] = data_conf['test']
    options['nc']  = data_conf['nc']
    options['names'] = data_conf['names']
    return options


if __name__ == '__main__':
    config_path = "../config/PCB.yaml"
    module_defs = parse_data_config(config_path)
    print(module_defs)
