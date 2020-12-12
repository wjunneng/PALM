# -*- coding:utf-8 -*-
import os
from examples.entity_linking.lib import lib_tools

lib_tools.seed_everything()

project_dir = '/'.join(os.path.abspath(__file__).split('/')[:-4])

import pandas as pd


class PATH(object):
    DATA_DIR = os.path.join(project_dir, 'examples', 'entity_linking', 'data')

    # -*- input -*-
    INPUT_DIR = os.path.join(DATA_DIR, 'input')
    OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
    LOG_DIR = os.path.join(DATA_DIR, 'log')
    MODEL_DIR = os.path.join(DATA_DIR, 'model')
    SUBMIT_DIR = os.path.join(DATA_DIR, 'submit')

    # -*- pretrain model -*-
    MODEL_ERNIE_PYTORCH = os.path.join(MODEL_DIR, 'model')

    # -*--*- 初赛 -*--*-
    input_train_json_path = os.path.join(INPUT_DIR, 'train.json')
    input_test_json_path = os.path.join(INPUT_DIR, 'test.json')
    input_dev_json_path = os.path.join(INPUT_DIR, 'dev.json')
    input_kb_json_path = os.path.join(INPUT_DIR, 'kb.json')


class CONFIG(object):
    PICKLE_DATA = {
        # 实体名称对应的KBID列表
        'ENTITY_TO_KBIDS': None,
        # KBID对应的实体名称列表
        'KBID_TO_ENTITIES': None,
        # KBID对应的属性文本
        'KBID_TO_TEXT': None,
        # KBID对应的实体类型列表（注意：一个实体可能对应'|'分割的多个类型）
        'KBID_TO_TYPES': None,
        # KBID对应的关系属性列表
        'KBID_TO_PREDICATES': None,
        # KBID对应的客体属性列表
        'KBID_TO_OBJECTS': None,

        # 索引类型映射列表
        'IDX_TO_TYPE': None,
        # 类型索引映射字典
        'TYPE_TO_IDX': None,
    }


for item in PATH.__dict__:
    if 'dir' in item.lower():
        if not os.path.exists(PATH().__getattribute__(item)):
            os.makedirs(PATH().__getattribute__(item))

for key in CONFIG.PICKLE_DATA:
    key_path = os.path.join(PATH.OUTPUT_DIR, key.lower() + '.pkl')
    if os.path.exists(key_path):
        CONFIG.PICKLE_DATA[key] = pd.read_pickle(path=key_path)
    else:
        print('{} not exists!!!'.format(key_path))
