# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from examples.entity_linking.config.config import project_dir, PATH, CONFIG
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

    def preprocess_entity_link(self, input_file_path, output_file_path):
        """
            处理entity linking 数据
        """
        df = pd.read_csv(input_file_path, encoding='utf-8', sep='\t')

        text_a_list, text_b_list, label_list = [], [], []
        for row_int in tqdm(range(df.shape[0])):
            # text_id	entity	offset	short_text	kb_id	kb_text	kb_predicate_num	predict
            text_id = df.iloc[row_int, 0]
            entity = df.iloc[row_int, 1]
            offset = df.iloc[row_int, 2]
            short_text = df.iloc[row_int, 3]
            kb_id = df.iloc[row_int, 4]
            kb_text = df.iloc[row_int, 5]
            kb_predicate_num = df.iloc[row_int, 6]
            predict = df.iloc[row_int, 7]

            # entity + ' ' + short_text
            text_a = entity + ' ' + short_text
            # kb_text
            text_b = kb_text
            # predict
            label = predict

            text_a_list.append(text_a)
            text_b_list.append(text_b)
            label_list.append(label)

        result = pd.DataFrame({'text_a': text_a_list, 'text_b': text_b_list, 'label': label_list})
        result.to_csv(path_or_buf=output_file_path, sep='\t', encoding='utf-8', index=None)

    def preprocess_entity_type(self, input_file_path, output_file_path):
        """
            处理entity type 数据
        """
        print(CONFIG.PICKLE_DATA['TYPE_TO_IDX'])
        df = pd.read_csv(input_file_path, encoding='utf-8', sep='\t', engine='python')

        text_a_list, text_b_list, label_list = [], [], []
        for row_int in tqdm(range(df.shape[0])):
            # text_id	entity	offset	short_text	type
            text_id = df.iloc[row_int, 0]
            entity = df.iloc[row_int, 1]
            offset = df.iloc[row_int, 2]
            short_text = df.iloc[row_int, 3]
            type = df.iloc[row_int, 4]

            # entity
            text_a = entity
            # kb_text
            text_b = short_text
            # type
            label = type

            text_a_list.append(text_a)
            text_b_list.append(text_b)
            label_list.append(label)

        result = pd.DataFrame({'text_a': text_a_list, 'text_b': text_b_list, 'label': label_list})
        result['label'] = result['label'].map(CONFIG.PICKLE_DATA['TYPE_TO_IDX'])
        result.to_csv(path_or_buf=output_file_path, sep='\t', encoding='utf-8', index=None)

    def main(self):
        # self.preprocess_entity_link(input_file_path=os.path.join(PATH.OUTPUT_DIR, 'el_train.tsv'),
        #                             output_file_path=os.path.join(PATH.OUTPUT_ENTITY_LINK_DIR, 'train.tsv'), )
        # self.preprocess_entity_link(input_file_path=os.path.join(PATH.OUTPUT_DIR, 'el_dev.tsv'),
        #                             output_file_path=os.path.join(PATH.OUTPUT_ENTITY_LINK_DIR, 'dev.tsv'), )
        # self.preprocess_entity_link(input_file_path=os.path.join(PATH.OUTPUT_DIR, 'el_test.tsv'),
        #                             output_file_path=os.path.join(PATH.OUTPUT_ENTITY_LINK_DIR, 'test.tsv'), )

        self.preprocess_entity_type(input_file_path=os.path.join(PATH.OUTPUT_DIR, 'et_train.tsv'),
                                    output_file_path=os.path.join(PATH.OUTPUT_ENTITY_TYPE_DIR, 'train.tsv'), )
        self.preprocess_entity_type(input_file_path=os.path.join(PATH.OUTPUT_DIR, 'et_dev.tsv'),
                                    output_file_path=os.path.join(PATH.OUTPUT_ENTITY_TYPE_DIR, 'dev.tsv'), )
        self.preprocess_entity_type(input_file_path=os.path.join(PATH.OUTPUT_DIR, 'et_test.tsv'),
                                    output_file_path=os.path.join(PATH.OUTPUT_ENTITY_TYPE_DIR, 'test.tsv'), )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DataFramePreprocessor...')

    args = parser.parse_args()
    DataFramePreprocessor(args=args).main()
