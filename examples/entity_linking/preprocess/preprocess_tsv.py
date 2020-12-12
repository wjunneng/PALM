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

from examples.entity_linking.config.config import CONFIG


class DataFramePreprocessor:
    """
        生成模型训练\验证\测试所需的tsv文件
    """

    def __init__(self, args):
        self.args = args

        self.train_max_negs = args.train_max_negs
        self.dev_max_negs = args.dev_max_negs
        self.test_max_negs = args.test_max_negs

    def preprocess_entity_link(self, input_file_path, output_file_path, max_negs):
        """
            处理entity linking 数据

            {'text_id': '52243',
            'text': '我之所以登山,是因为山在那里。这样一种',
            'mention_data': [{'kb_id': '85747', 'mention': '山', 'offset': '5'}]}
        """
        entity_to_kbids = CONFIG.PICKLE_DATA['ENTITY_TO_KBIDS']
        kbid_to_text = CONFIG.PICKLE_DATA['KBID_TO_TEXT']
        kbid_to_predicates = CONFIG.PICKLE_DATA['KBID_TO_PREDICATES']
        link_dict = defaultdict(list)

        with open(input_file_path, 'r') as file:
            for line in tqdm(file):
                line = json.loads(line)

                for mention in line['mention_data']:

                    # 处理测试集
                    if 'kb_id' not in mention:
                        mention['kb_id'] = '0'

                    # 如果是NIL,返回
                    if not mention['kb_id'].isdigit():
                        continue

                    entity = mention['mention']
                    kbids = list(entity_to_kbids[entity])
                    random.shuffle(kbids)

                    # 负样本， 此处构造的负样本是知识库中同一个实体(对应不同的含义)中不与当前正样本含义相同的样本.
                    num_negs = 0
                    for kbid in kbids:
                        if num_negs >= max_negs > 0 and kbid != mention['kb_id']:
                            continue

                        # ['1']
                        link_dict['text_id'].append(line['text_id'])
                        # ['小品']
                        link_dict['entity'].append(entity)
                        # ['0']
                        link_dict['offset'].append(mention['offset'])
                        # ['小品《战狼故事》中，吴京突破重重障碍解救爱人，深情告白太感人']
                        link_dict['short_text'].append(line['text'])
                        # ['275897']
                        link_dict['kb_id'].append(kbid)
                        # ['标签:语言、文学、曲艺、戏剧 基本要求:语言清晰，形态自然等 代表:喜剧小品 中文名:小品 摘要:小品，就是小的艺术品。
                        # 演艺注意事项:身心放松，自信 义项描述:小品 特点:短小精悍，情节简单等']
                        link_dict['kb_text'].append(kbid_to_text[kbid])
                        # [8]
                        link_dict['kb_predicate_num'].append(len(kbid_to_predicates[kbid]))

                        if kbid != mention['kb_id']:
                            link_dict['predict'].append(0)
                            num_negs += 1
                        else:
                            link_dict['predict'].append(1)

        line_df = pd.DataFrame(link_dict)
        line_df.to_csv(path_or_buf=output_file_path, index=None, sep='\t', encoding='utf-8')

    def preprocess_entity_type(self, input_file_path, output_file_path):
        """
            处理entity type 数据

            {'text_id': '52243',
            'text': '我之所以登山,是因为山在那里。这样一种',
            'mention_data': [{'kb_id': '85747', 'mention': '山', 'offset': '5'}]}
        """
        kbid_to_types = CONFIG.PICKLE_DATA['KBID_TO_TYPES']
        type_dict = defaultdict(list)

        with open(input_file_path, 'r') as file:
            for line in tqdm(file):
                line = json.loads(line)

                for mention in line['mention_data']:
                    entity = mention['mention']

                    # 测试集
                    if 'kb_id' not in mention:
                        entity_type = ['Other']
                    elif mention['kb_id'].isdigit():
                        entity_type = kbid_to_types[mention['kb_id']]
                    else:
                        entity_type = [i[4:] for i in mention['kb_id'].split('|')]

                    for e in entity_type:
                        type_dict['text_id'].append(line['text_id'])
                        type_dict['entity'].append(entity)
                        type_dict['offset'].append(mention['offset'])
                        type_dict['short_text'].append(line['text'])
                        type_dict['type'].append(e)

        type_df = pd.DataFrame(type_dict)
        type_df.to_csv(path_or_buf=output_file_path, index=None, encoding='utf-8', sep='\t')

    def main(self):
        self.preprocess_entity_link(input_file_path=os.path.join(PATH.input_train_json_path),
                                    output_file_path=os.path.join(PATH.OUTPUT_DIR, 'el_train.tsv'),
                                    max_negs=self.train_max_negs)
        self.preprocess_entity_link(input_file_path=os.path.join(PATH.input_dev_json_path),
                                    output_file_path=os.path.join(PATH.OUTPUT_DIR, 'el_dev.tsv'),
                                    max_negs=self.train_max_negs)
        self.preprocess_entity_link(input_file_path=os.path.join(PATH.input_test_json_path),
                                    output_file_path=os.path.join(PATH.OUTPUT_DIR, 'el_test.tsv'),
                                    max_negs=self.train_max_negs)

        self.preprocess_entity_type(input_file_path=os.path.join(PATH.input_train_json_path),
                                    output_file_path=os.path.join(PATH.OUTPUT_DIR, 'et_train.tsv'))
        self.preprocess_entity_type(input_file_path=os.path.join(PATH.input_dev_json_path),
                                    output_file_path=os.path.join(PATH.OUTPUT_DIR, 'et_dev.tsv'))
        self.preprocess_entity_type(input_file_path=os.path.join(PATH.input_test_json_path),
                                    output_file_path=os.path.join(PATH.OUTPUT_DIR, 'et_test.tsv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DataFramePreprocessor...')
    parser.add_argument('--train_max_negs', type=int, default=2, help='Train: the number of negative sample.')
    parser.add_argument('--dev_max_negs', type=int, default=-1, help='Dev: the number of negative sample.')
    parser.add_argument('--test_max_negs', type=int, default=-1, help='Test: the number of negative sample')

    args = parser.parse_args()
    DataFramePreprocessor(args=args).main()
