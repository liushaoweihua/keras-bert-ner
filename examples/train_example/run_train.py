# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
sys.path.append("../..")


import os
from keras_bert_ner.helper import train_args_parser
from keras_bert_ner.train import train


def run_train():
    # 基础配置
    args = train_args_parser()
    args.train_data = os.path.abspath(args.train_data)
    args.dev_data = os.path.abspath(args.dev_data)
    args.save_path = os.path.abspath(args.save_path)
    if True:
        param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
        print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))
    train(args=args)


if __name__ == '__main__':
    run_train()