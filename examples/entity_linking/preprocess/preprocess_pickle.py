# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from examples.entity_linking.config.config import project_dir, PATH
import os
import sys

sys.path.append(project_dir)
os.chdir(sys.path[-1])

import random

from examples.entity_linking.lib.lib_tools import seed_everything

seed_everything(seed=42)

import json
import argparse
import pandas as pd

from tqdm import tqdm
from collections import defaultdict


class PicklePreprocessor:
    """
        生成全局Pickle文件的预处理器
    """

    def __init__(self, args):
        # 实体名称对应的KBID列表，一个实体名可能对应多个不同的KBID，一词多义
        self.entity_to_kbids = defaultdict(set)
        # KBID对应的实体名称列表，一个KBID可能对应多个名称的实体，多词同义
        self.kbid_to_entities = dict()
        # KBID对应的属性文本 一个KBID对应一个属性文本
        self.kbid_to_text = dict()
        # KBID对应的实体类型列表
        self.kbid_to_types = dict()
        # KBID对应的关系属性列表
        self.kbid_to_predicates = dict()
        # KBID对应的客体属性列表
        self.kbid_to_objects = dict()

        # 索引类型映射列表
        self.idx_to_type = list()
        # 类型索引映射字典
        self.type_to_idx = dict()

        self.shuffle_text = args.shuffle_text

    def main(self):
        """
        {'alias': [],
        'subject_id': '122969',
        'data': [{'predicate': '摘要', 'object': '《青衣》是由江苏省文投集团策划、江苏大剧院出品制作的原创京剧现代戏。'},
                {'predicate': '义项描述', 'object': '京剧'}],
        'type': 'Culture',
        'subject': '青衣'}
        """
        with open(PATH.input_kb_json_path, 'r') as file:
            for line in tqdm(file):
                line = json.loads(line)

                kbid_int = line['subject_id']
                # 别名+实体名
                entities_set = set(line['alias'])
                entities_set.add(line['subject'])

                for entity_str in entities_set:
                    self.entity_to_kbids[entity_str].add(kbid_int)
                self.kbid_to_entities[kbid_int] = entities_set

                text_list, predicate_list, object_list = [], [], []
                for data_dict in line['data']:
                    predicate_str = data_dict['predicate'].strip()
                    object_str = data_dict['object'].strip()

                    text_list.append(':'.join([predicate_str, object_str]))
                    predicate_list.append(predicate_str)
                    object_list.append(object_str)
                if self.shuffle_text:
                    random.shuffle(text_list)
                self.kbid_to_text[kbid_int] = ' '.join(text_list)
                self.kbid_to_predicates[kbid_int] = predicate_list
                self.kbid_to_objects[kbid_int] = object_list
                # 删除特殊标记符号
                for special_symbol in ['\n', '\t', '\r']:
                    self.kbid_to_text[kbid_int] = self.kbid_to_text[kbid_int].replace(special_symbol, ' ')

                type_list = line['type'].split('|')
                self.kbid_to_types[kbid_int] = type_list
                for type_str in type_list:
                    if type_str not in self.type_to_idx:
                        self.type_to_idx[type_str] = len(self.idx_to_type)
                        self.idx_to_type.append(type_str)

        pd.to_pickle(obj=self.entity_to_kbids, path=os.path.join(PATH.OUTPUT_DIR, 'entity_to_kbids.pkl'))
        pd.to_pickle(obj=self.kbid_to_entities, path=os.path.join(PATH.OUTPUT_DIR, 'kbid_to_entities.pkl'))
        pd.to_pickle(obj=self.kbid_to_text, path=os.path.join(PATH.OUTPUT_DIR, 'kbid_to_text.pkl'))
        pd.to_pickle(obj=self.kbid_to_types, path=os.path.join(PATH.OUTPUT_DIR, 'kbid_to_types.pkl'))
        pd.to_pickle(obj=self.kbid_to_predicates, path=os.path.join(PATH.OUTPUT_DIR, 'kbid_to_predicates.pkl'))
        pd.to_pickle(obj=self.kbid_to_objects, path=os.path.join(PATH.OUTPUT_DIR, 'kbid_to_objects.pkl'))
        pd.to_pickle(obj=self.idx_to_type, path=os.path.join(PATH.OUTPUT_DIR, 'idx_to_type.pkl'))
        pd.to_pickle(obj=self.type_to_idx, path=os.path.join(PATH.OUTPUT_DIR, 'type_to_idx.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PicklePreprocessor...')
    parser.add_argument('--shuffle_text', action='store_true', default=False, help='shuffle or not')
    args = parser.parse_args()

    PicklePreprocessor(args=args).main()
