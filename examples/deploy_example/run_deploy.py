# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
sys.path.append("../..")


import os
import json
import time
import keras
import codecs
import tensorflow as tf
from loguru import logger
from termcolor import colored
from flask import Flask, Response, request
from keras_bert_ner.helper import deploy_args_parser
from keras_bert_ner.predict import build_trained_model, get_model_inputs


app = Flask(__name__)
app.model_configs = {}


def log_init(log_path):
    log_file_path = os.path.join(log_path, "info.log")
    err_file_path = os.path.join(log_path, "error.log")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger.add(sys.stderr, format="{time} {level} {message}",
               filter="my_module", level="INFO")
    logger.add(log_file_path, rotation="12:00", retention="14 days",
               encoding="utf-8")
    logger.add(err_file_path, rotation="100 MB", retention="14 days",
               encoding="utf-8", level="ERROR")
    logger.debug("logger initialized")
    return logger


def run_deploy():
    # 基础配置
    args = deploy_args_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_map if args.device_map != "cpu" else ""
    sess = tf.Session()
    graph = tf.get_default_graph()
    keras.backend.set_session(sess)
    # 属性设置
    with codecs.open(args.model_configs, "r", encoding="utf-8") as f:
        model_configs = json.load(f)
    args.max_len = model_configs.get("max_len")
    args.tag_padding = model_configs.get("tag_padding")
    args.model_path = os.path.abspath(model_configs.get("model_path"))
    args.bert_vocab = model_configs.get("bert_vocab")
    args.file_path = "/".join(args.model_path.split("/")[:-1])
    args.log_path = os.path.abspath(args.log_path)
    if True:
        param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
        print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))
    tokenizer, id_to_tag, viterbi_decoder = build_trained_model(args=args)
    # 新增属性
    app.model_configs["logger"] = log_init(args.log_path)
    app.model_configs["args"] = args
    app.model_configs["tokenizer"] = tokenizer
    app.model_configs["id_to_tag"] = id_to_tag
    app.model_configs["viterbi_decoder"] = viterbi_decoder
    return graph, sess


graph, sess = run_deploy()


def parse(text):
    # 编码
    token, seg = app.model_configs["tokenizer"].encode(text, first_length=app.model_configs["args"].max_len)
    # 解码
    # 显存不足时注释掉以下代码
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
    global graph, sess
    with graph.as_default():
        keras.backend.set_session(sess)
        decode_res = app.model_configs["viterbi_decoder"].decode([[token], [seg]])
        decode_res = [app.model_configs["id_to_tag"][item] for item in decode_res[0]
                      if app.model_configs["id_to_tag"][item] != app.model_configs["args"].tag_padding]
        min_len = min(app.model_configs["args"].max_len, len(text))
        decode_res = [text[:min_len], " ".join(decode_res[:min_len])]
        # decode_res = get_entities(decode_res)
        decode_res = json.dumps({"text": decode_res[0], "tags": decode_res[1]}, ensure_ascii=False, indent=4)
        return decode_res


# def get_entities(inputs):
#     text, tags = inputs
#     entities = []
#     tag = None
#     entity = ""
#     start, end = 0, 0
#     for i, text_item, tag_item in zip(range(len(text)), text, tags):
#         if tag_item.startswith("B"):
#             if tag is not None:
#                 entities.append({
#                     "start": start,
#                     "end": end,
#                     "entity_name": text[start:end],
#                     "entity_type": tag})
#             start, end = i, i + 1
#             entity = text_item
#             tag = tag_item.split("-")[-1] if tag_item.split("-")[-1] != "" else "unknown"
#         elif tag_item.startswith("I"):
#             if tag_item.split("-")[-1] in ["", tag]:
#                 entity += text_item
#                 end = i + 1
#             else:
#                 tag = None
#                 entity = ""
#                 start, end = i, i + 1
#         elif tag_item.startswith("S"):
#             if tag is not None:
#                 entities.append({
#                     "start": start,
#                     "end": i,
#                     "entity_name": text[start:i],
#                     "entity_type": tag})
#                 tag = None
#                 entity = ""
#                 start, end = i, i + 1
#             else:
#                 start, end = i, i + 1
#                 tag = tag_item.split("-")[-1] if tag_item.split("-")[-1] != "" else "unknown"
#                 entities.append({
#                     "start": start,
#                     "end": end,
#                     "entity_name": text[start:end],
#                     "entity_type": tag})
#                 tag = None
#                 entity = ""
#         else:
#             if tag is not None:
#                 entities.append({
#                     "start": start,
#                     "end": i,
#                     "entity_name": text[start:i],
#                     "entity_type": tag})
#                 tag = None
#                 entity = ""
#                 start, end = i, i + 1
#     if start != i and tag is not None:
#         entities.append({
#             "start": start,
#             "end": i,
#             "entity_name": text[start:i],
#             "entity_type": tag})
#     if entities == []:
#         return json.dumps({"text": text, "entities": None}, ensure_ascii=False, indent=4)
#     else:
#         return json.dumps({"text": text, "entities": entities}, ensure_ascii=False, indent=4)


def first_predict():
    """第一次使用模型时需要加载，会降低预测速度
    """
    parse("first_predict")


first_predict()


@app.route("/decode", methods=["POST"])
def decode():
    app.model_configs["logger"].info(colored("[RECEIVE]: ", "red") + colored(request.json["text"], "cyan"))
    res = parse(request.json["text"])
    app.model_configs["logger"].info(colored("[SEND]: ", "green") + colored(res, "cyan"))
    return res


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2020, debug=True, use_reloader=False)
