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

    # -----------------------  for training ----------------------- 

    # step 1-1: create readers for training
    match_reader = palm.reader.MatchReader(vocab_path, max_seqlen, seed=random_seed)
    # step 1-2: load the training data
    match_reader.load_data(train_file, file_format='tsv', num_epochs=num_epochs, batch_size=batch_size)

    # step 2: create a backbone of the model to extract text features
    ernie = palm.backbone.ERNIE.from_config(config)

    # step 3: register the backbone in reader
    match_reader.register_with(ernie)

    # step 4: create the task output head
    match_head = palm.head.Match(num_classes, input_dim, dropout_prob)

    # step 5-1: create a task trainer
    trainer = palm.Trainer(task_name)
    # step 5-2: build forward graph with backbone and task head
    loss_var = trainer.build_forward(ernie, match_head)

    # step 6-1*: use warmup
    n_steps = match_reader.num_examples * num_epochs // batch_size
    warmup_steps = int(0.1 * n_steps)
    print('total_steps: {}'.format(n_steps))
    print('warmup_steps: {}'.format(warmup_steps))
    sched = palm.lr_sched.TriangularSchedualer(warmup_steps, n_steps)

    # step 6-2: create a optimizer
    adam = palm.optimizer.Adam(loss_var, lr, sched)
    # step 6-3: build backward
    trainer.build_backward(optimizer=adam, weight_decay=weight_decay)

    # step 7: fit prepared reader and data
    trainer.fit_reader(match_reader)

    # step 8-1*: load pretrained parameters
    trainer.load_pretrain(pre_params, False)
    # step 8-2*: set saver to save model
    # save_steps = n_steps-16
    save_steps = 130000
    trainer.set_saver(save_path=save_path, save_steps=save_steps, save_type=save_type)
    # step 8-3: start training
    trainer.train(print_steps=print_steps)

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
