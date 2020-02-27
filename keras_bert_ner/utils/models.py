# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import keras
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras_contrib.layers import CRF
from .bert import *


def set_gelu(version):
    """设置gelu版本
    """
    version = version.lower()
    assert version in ["erf", "tanh"], "gelu version must be erf or tanh"
    if version == "erf":
        keras.utils.get_custom_objects()["gelu"] = gelu_erf
    else:
        keras.utils.get_custom_objects()["gelu"] = gelu_tanh


def gelu_erf(x):
    """基于Erf直接计算的gelu函数
    """
    return 0.5 * x * (1.0 + tf.math.erf(x / np.sqrt(2.0)))


def gelu_tanh(x):
    """基于Tanh近似计算的gelu函数
    """
    cdf = 0.5 * (1.0 + K.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * K.pow(x, 3)))))
    return x * cdf


set_gelu("tanh")


class NerBaseModel:
    """Bert Ner模型基础类
    """
    def __init__(self,
                 bert_config,
                 bert_checkpoint,
                 albert,
                 max_len,
                 numb_tags,
                 dropout_rate):
        self.bert_config = bert_config
        self.bert_checkpoint = bert_checkpoint
        self.albert = albert
        self.max_len = max_len
        self.numb_tags = numb_tags
        self.dropout_rate = dropout_rate
        self._build_bert_model()

    def _build_bert_model(self):
        """加载bert模型
        """
        self.bert_model = build_bert_model(
            self.bert_config,
            self.bert_checkpoint,
            albert=self.albert)
        for l in self.bert_model.layers:
            l.trainable = True

    def build(self):
        """Ner模型
        """
        x_in = Input(shape=(self.max_len,), name="Origin-Input-Token")
        s_in = Input(shape=(self.max_len,), name="Origin-Input-Segment")
        x = self.bert_model([x_in, s_in])
        x = Lambda(lambda X: X[:, 1:], name="Ignore-CLS")(x)
        x = self._task_layers(x)
        x = CRF(self.numb_tags, sparse_target=True, name="CRF")(x)
        model = Model([x_in, s_in], x)
        return model

    def _task_layers(self, layer):
        """下游网络层
        """
        raise NotImplementedError


class NerCnnModel(NerBaseModel):
    """Bert Ner模型 + Cnn下游模型
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 blocks,
                 *args,
                 **kwargs):
        super(NerCnnModel, self).__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.blocks = blocks

    def _task_layers(self, layer):
        def dilation_conv1d(dilation_rate, name):
            return Conv1D(self.filters, self.kernel_size, padding="same", dilation_rate=dilation_rate, name=name)

        def idcnn_block(name):
            return [dilation_conv1d(1, name + "1"), dilation_conv1d(1, name + "2"), dilation_conv1d(2, name + "3")]

        stack_layers = []
        for layer_idx in range(self.blocks):
            name = "Idcnn-Block-%s-Layer-" % layer_idx
            idcnns = idcnn_block(name)
            cnn = idcnns[0](layer)
            cnn = idcnns[1](cnn)
            cnn = idcnns[2](cnn)
            stack_layers.append(cnn)
        stack_layers = concatenate(stack_layers, axis=-1)
        return stack_layers


class NerRnnModel(NerBaseModel):
    """Bert Ner模型 + Rnn下游模型
    """
    def __init__(self,
                 cell_type,
                 units,
                 num_hidden_layers,
                 *args,
                 **kwargs):
        super(NerRnnModel, self).__init__(*args, **kwargs)
        self.cell_type = cell_type.lower()
        allowed_cell_type = ["lstm", "gru"]
        assert self.cell_type in allowed_cell_type, "cell_type must be one of %s" % allowed_cell_type
        self.units = units
        self.num_hidden_layers = num_hidden_layers

    def _task_layers(self, layer):
        if self.cell_type == "lstm":
            cell = LSTM
            cell_name = "Lstm"
        elif self.cell_type == "gru":
            cell = GRU
            cell_name = "Gru"
        else:
            raise ValueError("cell_type should be 'lstm' or 'gru'.")
        rnn = layer
        for layer_idx in range(self.num_hidden_layers):
            name = cell_name + "-%s" % layer_idx
            rnn = Bidirectional(
                cell(units=self.units, return_sequences=True, recurrent_dropout=self.dropout_rate), name=name)(rnn)
        return rnn