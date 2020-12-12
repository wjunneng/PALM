# -*- coding:utf-8 -*-
import os
import sys

sys.path.append(os.getcwd())

import json
import random
import pickle
import logging
import warnings
import numpy as np
import pandas as pd

from pathlib import Path

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')
# 设置显示的最大列数参数为a
pd.set_option('display.max_columns', 2000)
# 设置显示的最大的行数参数为b
pd.set_option('display.max_rows', 2000)
pd.set_option('display.width', 2000)

import logging

EPS = 1e-10


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    """
    初始化logger
    """
    if isinstance(log_file, Path):
        log_file = str(log_file)
    # log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    log_format = logging.Formatter("%(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def seed_everything(seed=42):
    """
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def kmax_pooling(x, dim, k):
    k = (x.size()[0] if x.size()[0] < k else k)
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]

    return x.gather(dim, index)


def default_load_csv(csv_file_path, encoding='utf-8', **kwargs):
    """
    加载csv文件
    :param csv_file_path:
    :param encoding:
    :param kwargs:
    :return:
    """
    tmp_df = pd.read_csv(csv_file_path, encoding=encoding, **kwargs)

    return tmp_df


def default_dump_csv(obj, csv_file_path, encoding='utf-8', **kwargs):
    """
    保存csv文件
    :param csv_file_path:
    :param encoding:
    :param kwargs:
    :return:
    """
    assert type(obj) == pd.DataFrame

    obj.to_csv(csv_file_path, encoding=encoding, **kwargs)


def default_load_json(json_file_path, encoding='utf-8', **kwargs):
    """
    加载json文件
    :param json_file_path:
    :param encoding:
    :param kwargs:
    :return:
    """
    with open(json_file_path, 'r', encoding=encoding) as fin:
        tmp_json = json.load(fin, **kwargs)
    return tmp_json


def default_dump_json(obj, json_file_path, encoding='utf-8', ensure_ascii=False, indent=2, **kwargs):
    """
    保存json文件
    :param obj:
    :param json_file_path:
    :param encoding:
    :param ensure_ascii:
    :param indent:
    :param kwargs:
    :return:
    """
    with open(json_file_path, 'w', encoding=encoding) as fout:
        json.dump(obj, fout, ensure_ascii=ensure_ascii, indent=indent, **kwargs)


def default_load_pkl(pkl_file_path, **kwargs):
    """
    加载pkl文件
    :param pkl_file_path:
    :param kwargs:
    :return:
    """
    with open(pkl_file_path, 'rb') as fin:
        obj = pickle.load(fin, **kwargs)

    return obj


def default_dump_pkl(obj, pkl_file_path, **kwargs):
    """
    保存pkl文件
    :param obj:
    :param pkl_file_path:
    :param kwargs:
    :return:
    """
    with open(pkl_file_path, 'wb') as fout:
        pickle.dump(obj, fout, **kwargs)


# 设置基础的日志配置
def set_basic_log_config():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)


def recursive_print_grad_fn(grad_fn, prefix='', depth=0, max_depth=50):
    if depth > max_depth:
        return
    print(prefix, depth, grad_fn.__class__.__name__)
    if hasattr(grad_fn, 'next_functions'):
        for nf in grad_fn.next_functions:
            ngfn = nf[0]
            recursive_print_grad_fn(ngfn, prefix=prefix + '  ', depth=depth + 1, max_depth=max_depth)


def strtobool(str_val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    str_val = str_val.lower()
    if str_val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif str_val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (str_val,))
