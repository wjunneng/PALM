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
import json
import numpy as np


def accuracy(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    return (preds == labels).mean()


def pre_recall_f1(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    # recall=TP/(TP+FN)
    tp = np.sum((labels == '1') & (preds == '1'))
    fp = np.sum((labels == '0') & (preds == '1'))
    fn = np.sum((labels == '1') & (preds == '0'))
    r = tp * 1.0 / (tp + fn)
    # Precision=TP/(TP+FP)
    p = tp * 1.0 / (tp + fp)
    epsilon = 1e-31
    f1 = 2 * p * r / (p + r + epsilon)
    return p, r, f1


def res_evaluate(res_dir="/home/wjunneng/Ubuntu/NLP/2020_12/PALM/outputs/predict/predictions.json", eval_phase='dev'):
    if eval_phase == 'test':
        data_dir = "/home/wjunneng/Ubuntu/NLP/2020_12/PALM/examples/entity_linking/data/output/entity_type/test.tsv"
    elif eval_phase == 'dev':
        data_dir = "/home/wjunneng/Ubuntu/NLP/2020_12/PALM/examples/entity_linking/data/output/entity_type/dev.tsv"
    else:
        assert eval_phase in ['dev', 'test'], 'eval_phase should be dev or test'

    labels = []
    with open(data_dir, "r") as file:
        first_flag = True
        for line in file:
            line = line.split("\t")
            label = line[2][:-1]
            if label == 'label':
                continue
            labels.append(str(label))
    file.close()

    preds = []
    with open(res_dir, "r") as file:
        for line in file.readlines():
            line = json.loads(line)
            pred = line['label']
            preds.append(str(pred))
    file.close()
    assert len(labels) == len(preds), "prediction result({}) doesn't match to labels({})".format(len(preds),
                                                                                                 len(labels))
    print('data num: {}'.format(len(labels)))
    p, r, f1 = pre_recall_f1(preds, labels)
    print("accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(accuracy(preds, labels), p, r, f1))


res_evaluate()
