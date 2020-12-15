# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# from examples.entity_linking.config.config import project_dir, PATH
# import os
# import sys
#
# sys.path.append(project_dir)
# os.chdir(sys.path[-1])
import os
import sys

project_dir = os.getcwd()
sys.path.append(project_dir)
os.chdir(sys.path[-1])

from examples.entity_linking.config.config import PATH

from examples.entity_linking.lib.lib_tools import seed_everything
from examples.entity_linking.core.evaluate_entity_link import res_evaluate

seed_everything(seed=42)

import paddlepalm as palm
import json
import os
import shutil

if __name__ == '__main__':
    # configs
    lr = 2e-5
    max_seqlen = 512
    batch_size = 4
    num_epochs = 3
    weight_decay = 0.0
    num_classes = 2
    random_seed = 1
    dropout_prob = 0.1

    save_type = 'ckpt'
    task_name = 'Quora Question Pairs matching'

    save_path = PATH.MODEL_ENTITY_LINK_DIR
    pred_output = PATH.OUTPUT_ENTITY_LINK_DIR
    pre_params = os.path.join(PATH.ERNIE_V2_CN_BASE_DIR, 'params')
    vocab_path = os.path.join(PATH.ERNIE_V2_CN_BASE_DIR, 'vocab.txt')
    config_path = os.path.join(PATH.ERNIE_V2_CN_BASE_DIR, 'ernie_config.json')

    dev_file = os.path.join(PATH.OUTPUT_ENTITY_LINK_DIR, 'dev.tsv')
    train_file = os.path.join(PATH.OUTPUT_ENTITY_LINK_DIR, 'train.tsv')
    predictions_dev_json_path = os.path.join(PATH.OUTPUT_ENTITY_LINK_DIR, 'predictions_dev.json')
    config = json.load(open(config_path))
    input_dim = config['hidden_size']

    # -----------------------  for training -----------------------
    print('do train...', flush=True)
    # step 1-1: create readers for training
    match_reader = palm.reader.MatchReader(vocab_path=vocab_path, max_len=max_seqlen, seed=random_seed)
    # step 1-2: load the training data
    match_reader.load_data(input_file=train_file, file_format='tsv', num_epochs=num_epochs, batch_size=batch_size)

    # step 2: create a backbone of the model to extract text features
    ernie = palm.backbone.ERNIE.from_config(config=config, phase='train')

    # step 3: register the backbone in reader
    match_reader.register_with(backbone=ernie)

    # step 4: create the task output head
    match_head = palm.head.Match(num_classes=num_classes, input_dim=input_dim, dropout_prob=dropout_prob)

    # step 5-1: create a task trainer
    trainer = palm.Trainer(name=task_name)
    # step 5-2: build forward graph with backbone and task head
    loss_var = trainer.build_forward(backbone=ernie, task_head=match_head)

    # step 6-1*: use warmup
    n_steps = match_reader.num_examples * num_epochs // batch_size
    save_steps = match_reader.num_examples // batch_size
    print('n_steps: {} save_steps: {}'.format(n_steps, save_steps), flush=True)  # n_steps: 401499 save_steps: 133833
    warmup_steps = int(0.1 * n_steps)
    sched = palm.lr_sched.TriangularSchedualer(warmup_steps=warmup_steps, num_train_steps=n_steps)

    # step 6-2: create a optimizer
    adam = palm.optimizer.Adam(loss_var=loss_var, lr=lr, lr_schedualer=sched)
    # step 6-3: build backward
    trainer.build_backward(optimizer=adam, weight_decay=weight_decay)

    # step 7: fit prepared reader and data
    iterator = trainer.fit_reader(reader=match_reader)

    # step 8-1*: load pretrained parameters
    trainer.load_pretrain(model_path=pre_params, convert=False)
    # step 8-2*: set saver to save model
    # save_steps = n_steps-16
    trainer.set_saver(save_path=save_path, save_steps=save_steps, save_type=save_type)
    # step 8-3: start training
    for step, batch in enumerate(iterator, start=1):
        one_step_result = trainer.train_one_step(batch=batch)
        if step % 1000 == 0:
            print('step: {} loss:{}'.format(step, list(one_step_result.values())[0]), flush=True)
        if step % (n_steps // 10) == 0:
            # -----------------------  for evaluation -----------------------
            print('do evaluation...')
            # step 1-1: create readers for training
            dev_match_reader = palm.reader.MatchReader(vocab_path=vocab_path, max_len=max_seqlen, seed=random_seed,
                                                       phase='predict')
            # step 1-2: load the training data
            dev_match_reader.load_data(dev_file, file_format='tsv', num_epochs=num_epochs, batch_size=batch_size)
            dev_num_examples = dev_match_reader.num_examples
            dev_print_steps = dev_num_examples // 1000
            # dev_num_examples: 142939 dev_print_steps: 143
            print('dev_num_examples: {} dev_print_steps: {}'.format(dev_num_examples, dev_print_steps), flush=True)

            # step 2: create a backbone of the model to extract text features
            dev_ernie = palm.backbone.ERNIE.from_config(config=config, phase='predict')

            # step 3: register the backbone in reader
            dev_match_reader.register_with(backbone=dev_ernie)

            # step 4: create the task output head
            dev_match_head = palm.head.Match(num_classes=num_classes, input_dim=input_dim, dropout_prob=dropout_prob,
                                             phase='predict', learning_strategy='pointwise')

            # step 5: build forward graph with backbone and task head
            trainer.build_predict_forward(pred_backbone=dev_ernie, pred_head=dev_match_head)

            # step 7: fit prepared reader and data
            trainer.fit_reader(reader=dev_match_reader, phase='predict')

            # step 8: predict
            trainer.predict(print_steps=dev_print_steps, output_dir=pred_output)

            shutil.copyfile(src=os.path.join(pred_output, 'predictions.json'),
                            dst=os.path.join(pred_output, 'predictions_dev.json'))
            os.remove(os.path.join(pred_output, 'predictions.json'))
            res_evaluate(res_dir=predictions_dev_json_path, eval_phase='dev')
