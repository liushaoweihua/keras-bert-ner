# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
import argparse


if os.name == "nt":
    bert_dir = ""
    root_dir = ""
else:
    bert_dir = "/home/liushaoweihua/pretrained_lm/bert_chinese/"
    root_dir = "/home/projects/Ner/tools/Keras-Bert-Ner/"


def train_args_parser():

    parser = argparse.ArgumentParser()

    data_group = parser.add_argument_group(
        "Data File Paths", "Config the train/dev/test file paths")
    data_group.add_argument("-train_data",
                            type=str,
                            default=os.path.join(root_dir, "data", "train.txt"),
                            required=True,
                            help="(REQUIRED) Train data path")
    data_group.add_argument("-dev_data",
                            type=str,
                            default=os.path.join(root_dir, "data", "dev.txt"),
                            help="Dev data path. Needed when -do_eval=True")

    save_group = parser.add_argument_group(
        "Model Output Paths", "Config the output paths for model")
    save_group.add_argument("-save_path",
                            type=str,
                            default=os.path.join(root_dir, "models"),
                            help="Model output paths")

    bert_group = parser.add_argument_group(
        "BERT File paths", "Config the path, checkpoint and filename of a pretrained or fine-tuned BERT model")
    bert_group.add_argument("-albert",
                            action="store_true",
                            default=False,
                            help="Whether to use ALBERT model. Default is False")
    bert_group.add_argument("-bert_config",
                            type=str,
                            default=os.path.join(bert_dir, "bert_config.json"),
                            required=True,
                            help="(REQUIRED) bert_config.json")
    bert_group.add_argument("-bert_checkpoint",
                            type=str,
                            default=os.path.join(bert_dir, "bert_model.ckpt"),
                            required=True,
                            help="(REQUIRED) bert_model.ckpt")
    bert_group.add_argument("-bert_vocab",
                            type=str,
                            default=os.path.join(bert_dir, "vocab.txt"),
                            required=True,
                            help="(REQUIRED) vocab.txt")

    action_group = parser.add_argument_group(
        "Action Configs", "Config the actions during running")
    action_group.add_argument("-do_eval",
                              action="store_true",
                              default=False,
                              help="Evaluation mode. Default is False")
    action_group.add_argument("-device_map",
                              type=str,
                              default="cpu",
                              help="Use CPU/GPU to train. If use CPU, then 'cpu'. "
                                   "If use GPU, then assign the devices, such as '0'. Default is 'cpu'")

    train_group = parser.add_argument_group(
        "Train Configs", "Config the train params")
    train_group.add_argument("-tag_padding",
                             type=str,
                             default="X",
                             help="Pad tag sequences less than max_len")
    train_group.add_argument("-best_fit",
                             action="store_true",
                             default=False,
                             help="Train best model that suits for dev.txt. Default is False")
    train_group.add_argument("-max_epochs",
                             type=int,
                             default=256,
                             help="Training epochs. Only available when -best_fit=True. Default is 256")
    train_group.add_argument("-early_stop_patience",
                             type=int,
                             default=3,
                             help="Early stop patience. Only available when -best_fit=True. Default is 3")
    train_group.add_argument("-reduce_lr_patience",
                             type=int,
                             default=2,
                             help="Reduce learning rate on plateau patience. "
                                  "Only available when -best_fit=True. Default is 2")
    train_group.add_argument("-reduce_lr_factor",
                             type=float,
                             default=0.5,
                             help="Reduce learning rate on plateau factor. "
                                  "Only available when -best_fit=True. Default is 0.5")
    train_group.add_argument("-hard_epochs",
                             type=int,
                             default=10,
                             help="Training epochs. Only available when -best_fit=False. Default is 10")
    train_group.add_argument("-batch_size",
                             type=int,
                             default=64,
                             help="Batch size. Default is 64")
    train_group.add_argument("-max_len",
                             type=int,
                             default=64,
                             help="Max sequence length. Default is 64")
    train_group.add_argument("-learning_rate",
                             type=float,
                             default=1e-5,
                             help="Initial adam lr. Default is 1e-5")

    model_group = parser.add_argument_group(
        "Model Configs", "Config the model params")
    model_group.add_argument("-model_type",
                             type=str,
                             default="rnn",
                             help="RNN models or CNN models. Default is rnn")
    model_group.add_argument("-cell_type",
                             type=str,
                             default="lstm",
                             help="Cell types. If model_type='rnn', could be 'lstm' or 'gru'. "
                                  "Default is lstm")
    model_group.add_argument("-rnn_units",
                             type=int,
                             default=128,
                             help="RNN units. "
                                  "Only available when model_type='rnn'. Default is 128")
    model_group.add_argument("-rnn_num_hidden_layers",
                             type=int,
                             default=1,
                             help="RNN hidden layer number. "
                                  "Only available when model_type='rnn'. Default is 1")
    model_group.add_argument("-cnn_filters",
                             type=int,
                             default=128,
                             help="CNN filters. Only available when model_type='cnn'. Default is 128")
    model_group.add_argument("-cnn_kernel_size",
                             type=int,
                             default=3,
                             help="CNN filters. Only available when model_type='cnn'. Default is 3")
    model_group.add_argument("-cnn_blocks",
                             type=int,
                             default=4,
                             help="IDCNN blocks. Only available when model_type='cnn'. Default is 4")
    model_group.add_argument("-dropout_rate",
                             type=float,
                             default=0.0,
                             help="Dropout rate. Default is 0.0")

    return parser.parse_args()


def predict_args_parser():

    parser = argparse.ArgumentParser()

    data_group = parser.add_argument_group(
        "Data File Paths", "Config the test file paths")
    data_group.add_argument("-test_data",
                            type=str,
                            default=os.path.join(root_dir, "data", "test.txt"),
                            required=True,
                            help="(REQUIRED) Test data path")

    model_group = parser.add_argument_group(
        "Model Paths", "Config the model paths")
    model_group.add_argument("-model_configs",
                             type=str,
                             default=os.path.join(root_dir, "models"),
                             required=True,
                             help="(REQUIRED) Model configs")

    output_group = parser.add_argument_group(
        "Output Paths", "Config the output paths")
    output_group.add_argument("-output_path",
                              type=str,
                              default=os.path.join(root_dir, "test_outputs"),
                              help="Output file paths")

    action_group = parser.add_argument_group(
        "Action Configs", "Config the actions during running")
    action_group.add_argument("-device_map",
                              type=str,
                              default="cpu",
                              help="Use CPU/GPU to train. "
                                   "If use CPU, then 'cpu'. "
                                   "If use GPU, then assign the devices, such as '0'. Default is 'cpu'")

    return parser.parse_args()


def deploy_args_parser():

    parser = argparse.ArgumentParser()

    model_group = parser.add_argument_group(
        "Model Paths", "Config the model paths")
    model_group.add_argument("-model_configs",
                             type=str,
                             default=os.path.join(root_dir, "models"),
                             required=True,
                             help="(REQUIRED) Model configs")

    logger_group = parser.add_argument_group(
        "Log Paths", "Config the log paths")
    logger_group.add_argument("-log_path",
                              type=str,
                              default=None,
                              required=True,
                              help="(REQUIRED) Log path")

    action_group = parser.add_argument_group(
        "Action Configs", "Config the actions during running")
    action_group.add_argument("-device_map",
                              type=str,
                              default="cpu",
                              help="Use CPU/GPU to train. "
                                   "If use CPU, then 'cpu'. "
                                   "If use GPU, then assign the devices, such as '0'. Default is 'cpu'")

    return parser.parse_args()


if __name__ == "__main__":
    parser_type = sys.argv[1].lower()
    if parser_type == "train":
        parser = train_args_parser()
    elif parser_type == "predict" or parser_type == "test":
        parser = predict_args_parser()
    elif parser_type == "deploy":
        parser = deploy_args_parser()
    parser.parse_args()