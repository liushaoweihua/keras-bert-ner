# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import json
import codecs
import pickle
import numpy as np
from keras.models import load_model
from keras_contrib.layers import CRF
from .utils.decoder import Viterbi
from .utils.models import gelu_erf, gelu_tanh
from .utils.metrics import CrfAcc, CrfLoss
from .utils.tokenizer import Tokenizer
from .utils.bert import MultiHeadAttention, LayerNormalization, PositionEmbedding, FeedForward, EmbeddingDense


def build_trained_model(args):
    """模型加载流程
    """
    # 环境设置
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_map if args.device_map != "cpu" else ""
    # 处理流程
    tokenizer = Tokenizer(args.bert_vocab)
    with codecs.open(os.path.join(args.file_path, "tag_to_id.pkl"), "rb") as f:
        tag_to_id = pickle.load(f)
    with codecs.open(os.path.join(args.file_path, "id_to_tag.pkl"), "rb") as f:
        id_to_tag = pickle.load(f)
    crf_accuracy = CrfAcc(tag_to_id, args.tag_padding).crf_accuracy
    crf_loss = CrfLoss(tag_to_id, args.tag_padding).crf_loss
    custom_objects = {
        "MultiHeadAttention": MultiHeadAttention,
        "LayerNormalization": LayerNormalization,
        "PositionEmbedding": PositionEmbedding,
        "FeedForward": FeedForward,
        "EmbeddingDense": EmbeddingDense,
        "CRF": CRF,
        "crf_accuracy": crf_accuracy,
        "crf_loss": crf_loss,
        "gelu_erf": gelu_erf,
        "gelu_tanh": gelu_tanh,
        "gelu": gelu_erf}
    model = load_model(args.model_path, custom_objects=custom_objects)
    model._make_predict_function()
    viterbi_decoder = Viterbi(model, len(id_to_tag))

    return tokenizer, id_to_tag, viterbi_decoder


def get_model_inputs(token_dict, texts, max_len):
    """获取模型的预测输入
    """
    tokenizer = Tokenizer(token_dict)
    tokens, segs = [], []
    for text in texts:
        token, seg = tokenizer.encode(text, first_length=max_len)
        tokens.append(np.array(token))
        segs.append(np.array(seg))

    return tokens, segs