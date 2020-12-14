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

import paddlepalm as palm
import json
import os
import shutil

if __name__ == '__main__':
    # configs
    max_seqlen = 512
    batch_size = 16
    lr = 2e-5
    num_classes = 24
    random_seed = 1
    print_steps = 100
    task_name = 'Quora Question Pairs matching'

    # 模式
    # mode = 'dev'
    mode = 'test'

    pred_output = PATH.OUTPUT_ENTITY_TYPE_DIR
    save_path = PATH.MODEL_ENTITY_TYPE_DIR
    vocab_path = os.path.join(PATH.ERNIE_V2_CN_BASE_DIR, 'vocab.txt')
    config_path = os.path.join(PATH.ERNIE_V2_CN_BASE_DIR, 'ernie_config.json')
    pre_params = os.path.join(PATH.ERNIE_V2_CN_BASE_DIR, 'params')
    train_file = os.path.join(PATH.OUTPUT_ENTITY_TYPE_DIR, 'train.tsv')
    predict_file = os.path.join(PATH.OUTPUT_ENTITY_TYPE_DIR, mode + '.tsv')

    config = json.load(open(config_path))
    input_dim = config['hidden_size']

    # -----------------------  for prediction -----------------------

    # step 1-1: create readers for prediction
    print('prepare to predict...')
    predict_match_reader = palm.reader.MatchReader(vocab_path, max_seqlen, seed=random_seed, phase='predict')
    # step 1-2: load the training data
    predict_match_reader.load_data(predict_file, batch_size)

    # step 2: create a backbone of the model to extract text features
    pred_ernie = palm.backbone.ERNIE.from_config(config, phase='predict')

    # step 3: register the backbone in reader
    predict_match_reader.register_with(pred_ernie)

    # step 4: create the task output head
    match_pred_head = palm.head.Match(num_classes, input_dim, phase='predict')

    # step 5-1: create a task trainer
    trainer = palm.Trainer(task_name)

    # step 5: build forward graph with backbone and task head
    trainer.build_predict_forward(pred_ernie, match_pred_head)

    # step 6: load checkpoint
    pred_model_path = os.path.join(PATH.MODEL_ENTITY_TYPE_DIR, 'ckpt.step200000')
    trainer.load_ckpt(pred_model_path)

    # step 7: fit prepared reader and data
    trainer.fit_reader(predict_match_reader, phase='predict')

    # step 8: predict
    print('predicting..')
    trainer.predict(print_steps=print_steps, output_dir=pred_output)

    shutil.copyfile(src=os.path.join(pred_output, 'predictions.json'),
                    dst=os.path.join(pred_output, 'predictions_' + mode + '.json'))

    os.remove(os.path.join(pred_output, 'predictions.json'))
