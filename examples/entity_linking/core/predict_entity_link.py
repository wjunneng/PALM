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

from examples.entity_linking.lib.lib_tools import seed_everything

seed_everything(seed=42)

import paddlepalm as palm
import json
import os

if __name__ == '__main__':
    # configs
    max_seqlen = 512
    batch_size = 4
    num_epochs = 3
    lr = 2e-5
    weight_decay = 0.0
    num_classes = 2
    random_seed = 1
    dropout_prob = 0.1
    save_type = 'ckpt'
    print_steps = 5000
    task_name = 'Quora Question Pairs matching'

    save_path = PATH.MODEL_ENTITY_LINK_DIR
    pred_output = PATH.OUTPUT_ENTITY_LINK_DIR
    vocab_path = os.path.join(PATH.ERNIE_V2_CN_BASE_DIR, 'vocab.txt')
    config_path = os.path.join(PATH.ERNIE_V2_CN_BASE_DIR, 'ernie_config.json')
    pre_params = os.path.join(PATH.ERNIE_V2_CN_BASE_DIR, 'params')

    train_file = os.path.join(PATH.OUTPUT_ENTITY_LINK_DIR, 'train.tsv')
    predict_file = os.path.join(PATH.OUTPUT_ENTITY_LINK_DIR, 'dev.tsv')
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
    pred_model_path = os.path.join(PATH.MODEL_ENTITY_LINK_DIR, 'ckpt.step130000')
    trainer.load_ckpt(pred_model_path)

    # step 7: fit prepared reader and data
    trainer.fit_reader(predict_match_reader, phase='predict')

    # step 8: predict
    print('predicting..')
    trainer.predict(print_steps=print_steps, output_dir=pred_output)
