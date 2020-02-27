# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
sys.path.append("../..")


import os
import json
import codecs
from keras_bert_ner.helper import predict_args_parser
from keras_bert_ner.predict import build_trained_model, get_model_inputs


def run_test():
    # 基础配置
    args = predict_args_parser()
    with codecs.open(args.model_configs, "r", encoding="utf-8") as f:
        model_configs = json.load(f)
    args.max_len = model_configs.get("max_len")
    args.tag_padding = model_configs.get("tag_padding")
    args.model_path = model_configs.get("model_path")
    args.bert_vocab = model_configs.get("bert_vocab")
    args.output_path = os.path.abspath(args.output_path)
    args.model_configs = os.path.abspath(args.model_configs)
    args.test_data = os.path.abspath(args.test_data)
    args.file_path = "/".join(args.model_path.split("/")[:-1])
    if True:
        param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
        print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))

    # 编码
    tokenizer, id_to_tag, viterbi_decoder = build_trained_model(args=args)
    with codecs.open(args.test_data, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    tokens, segs = get_model_inputs(args.bert_vocab, test_data, max_len=args.max_len)

    # 解码
    decode_res = viterbi_decoder.decode([tokens, segs])
    decode_res = [[id_to_tag[item] for item in term if id_to_tag[item] != args.tag_padding] for term in decode_res]
    decode_res = [[data_item[:min(args.max_len, len(data_item))], " ".join(decode_item[:min(args.max_len, len(data_item))])]
                  for data_item, decode_item in zip(test_data, decode_res)]

    # 保存
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    with codecs.open(os.path.join(args.output_path, "test_outputs.txt"), "w", encoding="utf-8") as f:
        json.dump(decode_res, f, indent=4, ensure_ascii=False)
    
if __name__ == "__main__":
    run_test()