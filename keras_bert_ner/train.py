# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import time
import json
import keras
import codecs
import pickle
import numpy as np
from .utils.tokenizer import Tokenizer
from .utils.processor import Processor
from .utils.models import NerCnnModel, NerRnnModel
from .utils.callbacks import NerCallbacks
from .utils.metrics import CrfAcc, CrfLoss


def train(args):
    """模型训练流程
    """
    # 环境设置
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_map if args.device_map != "cpu" else ""
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # 数据准备
    processor = Processor(args.train_data, args.bert_vocab, args.tag_padding)
    train_tokens, train_segs, train_tags = processor.process(args.train_data, args.max_len)
    train_x = [np.array(train_tokens), np.array(train_segs)]
    train_y = np.array(train_tags)
    if args.do_eval:
        dev_tokens, dev_segs, dev_tags = processor.process(args.dev_data, args.max_len)
        devs = [[np.array(dev_tokens), np.array(dev_segs)], np.array(dev_tags)]
    else:
        devs = None
    # 模型准备
    if args.model_type == "cnn":
        ner_model = NerCnnModel(
            bert_config=args.bert_config,
            bert_checkpoint=args.bert_checkpoint,
            albert=args.albert,
            max_len=args.max_len,
            numb_tags=processor.numb_tags,
            dropout_rate=args.dropout_rate,
            filters=args.cnn_filters,
            kernel_size=args.cnn_kernel_size,
            blocks=args.cnn_blocks).build()
    elif args.model_type == "rnn":
        ner_model = NerRnnModel(
            bert_config=args.bert_config,
            bert_checkpoint=args.bert_checkpoint,
            albert=args.albert,
            max_len=args.max_len,
            numb_tags=processor.numb_tags,
            dropout_rate=args.dropout_rate,
            cell_type=args.cell_type,
            units=args.rnn_units,
            num_hidden_layers=args.rnn_num_hidden_layers).build()
    else:
        raise ValueError("model_type should be 'cnn' or 'rnn'.")
    crf_accuracy = CrfAcc(processor.tag_to_id, args.tag_padding).crf_accuracy
    crf_loss = CrfLoss(processor.tag_to_id, args.tag_padding).crf_loss
    ner_model.compile(
        optimizer=keras.optimizers.Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
        loss=crf_loss,
        metrics=[crf_accuracy])
    # 模型训练
    bert_type = "ALBERT" if args.albert else "BERT"
    model_type = "IDCNN-CRF" if args.model_type == "cnn" else ("BILSTM-CRF" if args.cell_type == "lstm" else "BIGRU-CRF")
    model_save_path = os.path.abspath(
        os.path.join(args.save_path, "%s-%s.h5" % (bert_type, model_type)))
    log_save_path = os.path.abspath(
        os.path.join(args.save_path, "%s-%s-%s.log" % (bert_type, model_type, time.strftime("%Y%m%d_%H%M%S"))))
    if args.best_fit:
        best_fit_params = {
            "early_stop_patience": args.early_stop_patience,
            "reduce_lr_patience": args.reduce_lr_patience,
            "reduce_lr_factor": args.reduce_lr_factor,
            "save_path": model_save_path
        }
        callbacks = NerCallbacks(processor.id_to_tag, best_fit_params, args.tag_padding, log_save_path)
        epochs = args.max_epochs
    else:
        callbacks = NerCallbacks(processor.id_to_tag, None, args.tag_padding, log_save_path)
        epochs = args.hard_epochs

    ner_model.fit(
        x=train_x,
        y=train_y,
        batch_size=args.batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=devs)

    # 保存信息
    with codecs.open(os.path.join(args.save_path, "tag_to_id.pkl"), "wb") as f:
        pickle.dump(processor.tag_to_id, f)
    with codecs.open(os.path.join(args.save_path, "id_to_tag.pkl"), "wb") as f:
        pickle.dump(processor.id_to_tag, f)
    model_configs = {
        "max_len": args.max_len,
        "tag_padding": args.tag_padding,
        "model_path": model_save_path,
        "bert_vocab": os.path.abspath(args.bert_vocab)}
    with codecs.open(os.path.join(args.save_path, "model_configs.json"), "w") as f:
        json.dump(model_configs, f, ensure_ascii=False, indent=4)
    if not args.best_fit:
        ner_model.save(model_save_path)