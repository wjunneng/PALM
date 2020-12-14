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

import json
import random
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser

from examples.entity_linking.lib.lib_tools import seed_everything
from examples.entity_linking.lib import lib_tools

seed_everything(seed=42)


class Submit(object):
    def __init__(self, args):
        self.args = args
        self.entity_link_test_tsv_path = os.path.join(PATH.OUTPUT_DIR, 'el_test.tsv')
        self.entity_type_test_tsv_path = os.path.join(PATH.OUTPUT_DIR, 'et_test.tsv')

        self.entity_link_predictions_json_path = os.path.join(PATH.OUTPUT_ENTITY_LINK_DIR, 'predictions_test.json')
        self.entity_type_predictions_json_path = os.path.join(PATH.OUTPUT_ENTITY_TYPE_DIR, 'predictions_test.json')
        self.submit_json = os.path.join(PATH.SUBMIT_DIR, 'result.json')

    def main(self):
        entity_link_list, entity_type_list = [], []

        with open(self.entity_link_predictions_json_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                line = json.loads(line)
                entity_link_list.append(line['label'])

        with open(self.entity_type_predictions_json_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                line = json.loads(line)
                entity_type_list.append(line['label'])

        entity_link_df = pd.read_csv(filepath_or_buffer=self.entity_link_test_tsv_path, sep='\t')
        entity_link_df = entity_link_df[['text_id', 'entity', 'offset', 'short_text', 'kb_id']]
        entity_link_df['entity_link'] = entity_link_list

        entity_type_df = pd.read_csv(filepath_or_buffer=self.entity_type_test_tsv_path, sep='\t')
        entity_type_df = entity_type_df[['text_id', 'entity', 'offset', 'short_text', 'type']]
        entity_type_df['entity_type'] = entity_type_list
        entity_type_df['entity_type'] = entity_type_df['entity_type'].map(
            {value: key for (key, value) in CONFIG.PICKLE_DATA['TYPE_TO_IDX'].items()})

        df = entity_link_df.merge(entity_type_df, on=('text_id', 'offset'), how='outer', suffixes=('', '_et'))

        df_ = df.iloc[entity_link_df.shape[0]:]
        df_['entity'] = df_['entity_et']
        df_['short_text'] = df_['short_text_et']
        df_ = df_[['text_id', 'entity', 'offset', 'short_text', 'kb_id', 'entity_link', 'entity_type']]
        df_['kb_id'] = -1
        df_['entity_link'] = 0

        df = df.loc[:entity_link_df.shape[0] - 1]
        df = df[['text_id', 'entity', 'offset', 'short_text', 'kb_id', 'entity_link', 'entity_type']]

        df = pd.merge(df, df_, how='outer', on=['text_id', 'entity', 'offset', 'short_text', 'kb_id', 'entity_link', 'entity_type'])
        df.to_csv('result_.csv', encoding='utf-8', index=None)
        df['text_id'] = df['text_id'].astype(int)
        df.sort_values(by=['text_id', 'offset'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        df['entity_link'] = df['entity_link'].astype(int)
        df['kb_id'] = df['kb_id'].astype(int)

        sample_list = []
        for (text_id_int, text_id_df) in df.groupby(by=['text_id']):
            sample_dict = {}
            sample_dict['text_id'] = str(text_id_int)
            sample_dict['text'] = text_id_df.iloc[0, 3]
            mention_data_list = []

            for (offset_int, offset_df) in text_id_df.groupby(by=['offset']):
                mention_sample_dict = {}

                offset_df.sort_values(by=['entity_link'], ascending=False, inplace=True)

                if offset_df.shape[0] >= 3:
                    print(offset_df)

                if int(offset_df.iloc[0, -2]) == 1:
                    mention_sample_dict['kb_id'] = str(offset_df.iloc[0, 4])
                elif int(offset_df.iloc[0, -2]) == 0:
                    mention_sample_dict['kb_id'] = 'NIL_' + str(offset_df.iloc[0, -1])
                assert int(offset_df.iloc[0, -2]) in [0, 1]

                mention_sample_dict['mention'] = offset_df.iloc[0, 1]
                mention_sample_dict['offset'] = str(offset_int)
                mention_data_list.append(mention_sample_dict)

            sample_dict['mention_data'] = mention_data_list

            sample_list.append(sample_dict)

        os.remove(path=self.submit_json)

        with open(self.submit_json, 'a', encoding='utf-8') as file:
            for sample_dict in sample_list:
                json.dump(sample_dict, file, ensure_ascii=False)
                file.write('\n')

        print(entity_link_df.head(20))
        print(entity_type_df.head(10))
        print(df.head(100))


if __name__ == '__main__':
    parser = ArgumentParser(description='Submit...')
    args = parser.parse_args()

    Submit(args=args).main()
