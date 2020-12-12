# -*- coding:utf-8 -*-
import argparse
import matplotlib.pyplot as plt


def draw_plot(x1, x2, y1, y2):
    """
    绘制折线图
    :return:
    """
    plt.plot(x1, y1, 'r--', label='train')
    plt.plot(x2, y2, 'g--', label='test')

    plt.plot(x1, y1, 'ro-', x2, y2)
    plt.xlabel('lenght')
    plt.ylabel('count')
    plt.legend()
    plt.show()


def get_run_parser() -> argparse.ArgumentParser:
    """
    return basic arg parser
    """
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--data_dir", type=str, help="data dir")
    parser.add_argument("--bert_config_dir", type=str, help="bert config dir")
    parser.add_argument("--data_cache", action='store_true', help="whether to cache data or not")
    parser.add_argument("--pretrained_checkpoint", default="", type=str, help="pretrained checkpoint path")
    parser.add_argument("--max_length", type=int, default=300, help="max length of dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=1, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps used for scheduler.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

    return parser


def get_infer_parser() -> argparse.ArgumentParser:
    """
    return basic arg parser
    """
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--ckpt_path", type=str, help="ckpt path")
    parser.add_argument("--hparams_path", type=str, help="hparams path")
    parser.add_argument("--max_length", type=int, default=420, help="max length of dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--map_location", type=int, default=None, help="map location")
    parser.add_argument("--workers", type=int, default=None, help="workers")

    return parser


def remove_overlap(spans):
    """
    remove overlapped spans greedily for flat-ner
    Args:
        spans: list of tuple (start, end), which means [start, end] is a ner-span
    Returns:
        spans without overlap
    """
    output = []
    occupied = set()
    for start, end in spans:
        if any(x for x in range(start, end + 1)) in occupied:
            continue
        output.append((start, end))
        for x in range(start, end + 1):
            occupied.add(x)
    return output
