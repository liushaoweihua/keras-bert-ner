# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import codecs
import numpy as np
from keras.utils import to_categorical
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from .decoder import Viterbi


def NerCallbacks(id_to_tag, best_fit_params=None, mask_tag=None, log_path=None):
    """模型训练过程中的回调函数
    """
    callbacks = [Accuracy(id_to_tag, mask_tag, log_path)]
    if best_fit_params is not None:
        early_stopping = EarlyStopping(
            monitor="val_crf_accuracy",
            patience=best_fit_params.get("early_stop_patience"))
        reduce_lr_on_plateau = ReduceLROnPlateau(
            monitor="val_crf_accuracy", verbose=1, mode="max",
            factor=best_fit_params.get("reduce_lr_factor"),
            patience=best_fit_params.get("reduce_lr_patience"))
        model_check_point = ModelCheckpoint(
            best_fit_params.get("save_path"),
            monitor="val_crf_accuracy", verbose=2, mode="max", save_best_only=True)
        callbacks.extend([early_stopping, reduce_lr_on_plateau, model_check_point])
    return callbacks


class Accuracy(Callback):

    def __init__(self, id_to_tag, mask_tag=None, save_path=None):
        self.id_to_tag = id_to_tag
        self.mask_tag = mask_tag
        self.save_path = save_path
        self.numb_tags = len(self.id_to_tag)
        self.mask_tag_id = {v: k for k, v in self.id_to_tag.items()}.get(self.mask_tag)

    def on_epoch_end(self, epoch, logs=None):
        viterbi = Viterbi(self.model, self.numb_tags)
        val_true = np.squeeze(self.validation_data[2], axis=-1)
        mask = np.array(1. - to_categorical(val_true, self.numb_tags)[:, :, self.mask_tag_id]) \
            if self.mask_tag_id else None
        val_pred = viterbi.decode([self.validation_data[0], self.validation_data[1]])
        self._call_acc(val_true, val_pred, mask, epoch)

    def _call_acc(self, val_true, val_pred, mask, epoch):
        total_sentence_numb = val_true.shape[0]
        right_sentence_numb = 0
        total_tag_numb_dict = {tag: 0 for tag in self.id_to_tag.values() if tag != self.mask_tag}
        right_tag_numb_dict = {tag: 0 for tag in self.id_to_tag.values() if tag != self.mask_tag}
        if mask is None:
            for sentence_true, sentence_pred in zip(val_true, val_pred):
                if all(sentence_true == sentence_pred):
                    right_sentence_numb += 1
                for tag_true, tag_pred in zip(sentence_true, sentence_pred):
                    if tag_true == tag_pred:
                        right_tag_numb_dict[self.id_to_tag[tag_pred]] += 1
                    total_tag_numb_dict[self.id_to_tag[tag_true]] += 1
        else:
            for sentence_true, sentence_pred, sentence_mask in zip(val_true, val_pred, mask):
                if all(sentence_true * sentence_mask == sentence_pred * sentence_mask):
                    right_sentence_numb += 1
                for tag_true, tag_pred, tag_mask in zip(sentence_true, sentence_pred, sentence_mask):
                    if tag_mask == 0.:
                        continue
                    if tag_true == tag_pred:
                        right_tag_numb_dict[self.id_to_tag[tag_pred]] += 1
                    total_tag_numb_dict[self.id_to_tag[tag_true]] += 1
        sentence_acc = right_sentence_numb / total_sentence_numb
        tag_acc = {tag: right_tag_numb_dict[tag] / total_tag_numb_dict[tag] for tag in right_tag_numb_dict}
        callback_info = "*" * 30 + " Epoch " + str(epoch) + " " + "*" * 30 + "\n" \
                        + "-" * 25 + " Sentence Accuracy " + "-" * 25 + "\n" \
                        + "\t" * 2 + "Right" + "\t" * 2 + "Total" + "\t" * 2 + "Acc" + "\n" \
                        + "\t" * 2 + str(right_sentence_numb) + "\t" * 2 + str(total_sentence_numb) + "\t" * 2 + str(sentence_acc) + "\n" \
                        + "-" * 28 + " Tag Accuracy " + "-" * 27 + "\n" \
                        + "\t" * 2 + "Right" + "\t" * 2 + "Total" + "\t" * 2 + "Acc" + "\n"
        for tag in tag_acc:
            callback_info += tag + "\t" * 2 \
                          + str(right_tag_numb_dict[tag]) + "\t" * 2 \
                          + str(total_tag_numb_dict[tag]) + "\t" * 2 \
                          + str(tag_acc[tag]) + "\n"
        print(callback_info)
        if self.save_path is not None:
            with codecs.open(self.save_path, "a", encoding="utf-8") as f:
                f.write(callback_info)