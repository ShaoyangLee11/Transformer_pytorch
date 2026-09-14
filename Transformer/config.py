from pathlib import Path
import yaml

def get_config()->dict:
    return {
        "batch_size":8,
        "num_epochs":50,
        "lr":1e-4,
        "seq_len":350,
        "d_model":512,
        "lang_src":"en",
        "lang_tgt":"it",
        "model_folder":"weights",
        "model_basename":"tmodel_",
        "preload":None,
        "tokenizer_file":"tokenizer_{0}.json",
        "experiment_name":"runs/tmodel",
        "datasource":'Helsinki-NLP/opus_books'

    }

def get_config_yaml(filename:str)->dict:

    file1=open(file=filename,mode='r')
    config_str=file1.read()
    file1.close()

    yaml.load(config_str,Loader=yaml.FullLoader)

    return config_str


def get_weight_file(config,epoch:str)->str:

    model_folder=config['model_folder']
    model_basename=config['model_basename']
    model_filename=f'{model_basename}{epoch}.pt'

    return str(Path('.') / model_folder / model_filename)


"""
1. Path('.')
创建 Path 对象，代表当前工作目录: .
2. Path('.') / model_folder

重点：/ 是 pathlib.Path 类重载的除法运算符，不是除法！
作用：拼接路径片段。
   - model_folder 只是普通字符串(比如"weights")
   - Path对象 / 字符串 → 返回新的 Path 对象
   例子:Path('.') / "weights" → 代表 ./weights 的路径对象
3. 再接 / model_filename
继续拼接文件名，依然是 Path 对象之间的路径拼接。

整段 Path('.') / model_folder / model_filename 的结果：一个 Path 路径对象，不是字符串。
代码里写的 / 只是语法符号，这个符号本身不会进入最终字符串。

4. str(Path对象)
把路径对象转为普通字符串.

"""

def latest_weights_file_path(config):
  
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}*.pt"
    weights_files = list(Path(model_folder).glob(model_filename))

    if len(weights_files) == 0:
        return None
  
    weights_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
    return str(weights_files[-1])


    """
     .glob(model_filename): glob = 通配符查找，在这个文件夹里按文件名模式匹配文件

        - model_filename 支持通配符：
        - "*.pt"：匹配文件夹下面所有后缀是.pt的文件
        - "model_*.pt"：匹配 model_001.pt、model_100.pt
        - "model.pt"：精确找名叫 model.pt 的文件
    
    """