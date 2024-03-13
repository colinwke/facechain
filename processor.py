# coding=utf-8
# ----------------------------------------------------------------
# @time: 2024/03/12 15:53
# @author: Wang Ke
# @contact: wangke09@58.com
# ----------------------------------------------------------------
""" 为用户提供容器内部的python版预处理, 后处理, 自定义模型加载, 自定义模型执行功能
refs:
    1. http://cloud.58corp.com/doc/5b9f7798638a3b018b66d7dd/5db2577395263920412d9655
    2. 103 /opt/users/zhukaiwen01/workspace/git/FunASR-main/my_workspace/processor.py
"""


def preprocess(x, **kwargs):
    """ 预处理函数
    params: x:      用户 jar 包中, 数据处理类的 predictOnlineBefore 函数封装的数据, 类型包括(str, bytes, numpy.array, kwargs)
            kwargs: 用户 jar 包中, 数据处理类的 predictOnlineBefore 函数封装的参数
    return: 模型执行的输入数据
    """
    return x


def postprocess(x, **kwargs):
    """ 后处理函数
    params: x:       模型执行后的输出数据, 即`model(data)`所得得结果
            kwargs:  用户 jar 包中, 数据处理类的 predictOnlineBefore 函数封装的参数
    return: 用户 jar 包中, 数据处理类的 predictOnlineAfter 函数的输入数据类型
    """
    return x


def load_model():
    """ 模型加载函数, 用户自定义
    return: 加载好的模型, 用于模型推理
    """
    model = torch.load(model_path)
    return model


def run_model(model, x, **kwargs):
    """ 自定义推理执行函数
    params: model:  模型对象
            x:      预处理后的数据, 即 preprocess 函数所得得结果
            kwargs: 用户 jar 包中, 数据处理类的 predictOnlineBefore 函数封装的参数
    return: 模型推理处理结果
    """
    return model(x)
