[English Version](https://github.com/liushaoweihua/keras-bert-ner/blob/master/README.md)  |  [中文版说明](https://github.com/liushaoweihua/keras-bert-ner/blob/master/README_ZH.md)

# Keras-Bert-Ner

**中文命名实体识别任务**下的Keras解决方案，下游模型支持BiLSTM-CRF/BiGRU-CRF/IDCNN-CRF，预训练语言模型采用BERT系列：支持**BERT/RoBERTa/ALBERT**。

## 更新日志

* **2020.02.27** 重构`keras_bert_ner`代码，移除部分冗余文件。`bert4keras == 0.2.5`现在被集成在该项目中。

* **2019.11.14** 因为`bert4keras`的稳定性被应用于该项目的重要支撑模块。现在仅支持Google版本的**albert模型**。

* **2019.11.04** 修复预测过程中，句级别精度计算错误的问题。

* **2019.11.01** 采用自定义的crf_accuracy/crf_loss替换`keras-contrib`中的crf_accuracy/crf_loss，以处理**mask**标签的情况。

## 未来工作

* 迁移至tensorflow 2.0。

* 新增BERT系列改进模型，如Distill Bert，Tiny Bert。

## 依赖项

* flask == 1.1.1
* keras == 2.3.1
* numpy == 1.18.1
* loguru == 0.4.1
* requests == 2.22.0
* termcolor == 1.1.0
* tensorflow == 1.15.2
* keras_contrib == 2.0.8

## 训练阶段

> **数据形式**

```json
[
    [
        "揭秘趣步骗局，趣步是什么，趣步是怎么赚钱的？趣步公司可靠吗？趣步合法吗？相信是众多小伙伴最关心的话题，今天小编就来给大家揭开趣步这面“丑恶”且神秘的面纱，让小伙伴们看清事情的真相。接下来，我用简单的文字，给大家详细剖析一下趣步公司及趣步app的逻辑到底是什么样>的？3分钟时间...全文：?揭秘趣步骗局，趣步是什么，趣步是怎么赚钱的？趣步公司可靠吗？趣步合法吗？相信是众多小伙伴最关心的话题，今天小编就来给大家揭开趣步这面“丑恶”且神秘的面纱，让小伙伴们看清事情的真相。接下来，我用简单的文字，给大家详细剖析一下趣步公司及趣步app的逻>辑到底是什么样的？3分钟时间...全文：",
        "O O B I O O O B I O O O O B I O O O O O O O B I O O O O O O B I O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B I O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B I O O O B I O O O O O O O O O O O O O O O O O O O O O O O O O O O O B I O O O B I O O O O B I O O O O O O O B I O O O O O O B I O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B I O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B I O O O B I O O O O O O O O O O O O O O O O O O O O O O O O O"
    ],
    [
        "企业纳税贷额度，全国小微企业都可做！公司张总说：“没想到缴税还能办贷款，本来我们还在为准备纳税证明、银行流水而焦躁，这下好了，3分钟解决我们燃眉之急，这次当第一个吃螃蟹的人可真值得！”至今，产品上线推广两月有余，今天正式介绍一下这款主要满足小微企业生产经营过>程中真实合法的流动资金需求的“纳税贷”产品！一、产品特点1、贷款额度：最高300万2、贷款期限：12个月3、贷款利率：最低月息4厘4、贷款类型：免抵押、免担保、纯信用5、还款方式：后息后本6、申请方式：线上申请，无需提供纸质材料。二，准入条件：1、企业法人，企业主为中国内地公民，年龄18-65周岁之间2、企业生产经营1年以上，企业及企业主信用状况良好3、融资银行总计不超过3家(低风险及小额网贷业务除外）4、纳税等级A、B、C级，纳税总额2万以上，诚信纳税，无欠缴税款想体验“纳税贷”吗？欢迎加西部助贷。我们团队将为你提供全方位、定制化的服务。百万批款，3分钟>完成若你有银行贷款需求，但不符合这个产品的要求，请添加专员的个人微信进行咨询其他产品。我们会根据你的具体条件为你综合策划与优化，匹配申请其他低成本的产品，为您解决资金周转需求，欢迎咨询~西部助贷是伙伴领域资本旗下专业的助贷平台，专注于西部地区贷款研究，主营个人及中小微企业融资贷款金融咨询服务。金融团队秉承全心全意微企业服务的理念服务客户，我们团队服务的客户80%都是慕名而来与老客户转介绍！服务信得过！欢迎您咨询！我是西部助贷，主营个人/中小企业贷款金融服务，如您有贷款需求，欢迎拨打全国统一客服热线40087-90508，随时咨询更多最新信息。也欢迎您把这篇文章转发给身边有需要的朋友。",
        "O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B I I I O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B I I I O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B I I I O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O"
    ],
...
]
```

见`./examples/data/train.txt`，数据来源：[互联网金融新实体发现](https://www.datafountain.cn/competitions/361)

> **训练参数**

运行`python keras_bert_ner/helper.py train`，快速浏览训练参数。

```bash
(nlp) liushaoweihua@ai-server-8:~/projects/Ner/tools/Keras-Bert-Ner$ python keras_bert_ner/helper.py train
usage: helper.py [-h] -train_data TRAIN_DATA [-dev_data DEV_DATA]
                 [-save_path SAVE_PATH] [-albert] -bert_config BERT_CONFIG
                 -bert_checkpoint BERT_CHECKPOINT -bert_vocab BERT_VOCAB
                 [-do_eval] [-device_map DEVICE_MAP]
                 [-tag_padding TAG_PADDING] [-best_fit]
                 [-max_epochs MAX_EPOCHS]
                 [-early_stop_patience EARLY_STOP_PATIENCE]
                 [-reduce_lr_patience REDUCE_LR_PATIENCE]
                 [-reduce_lr_factor REDUCE_LR_FACTOR]
                 [-hard_epochs HARD_EPOCHS] [-batch_size BATCH_SIZE]
                 [-max_len MAX_LEN] [-learning_rate LEARNING_RATE]
                 [-model_type MODEL_TYPE] [-cell_type CELL_TYPE]
                 [-rnn_units RNN_UNITS]
                 [-rnn_num_hidden_layers RNN_NUM_HIDDEN_LAYERS]
                 [-cnn_filters CNN_FILTERS] [-cnn_kernel_size CNN_KERNEL_SIZE]
                 [-cnn_blocks CNN_BLOCKS] [-dropout_rate DROPOUT_RATE]
helper.py: error: the following arguments are required: -train_data, -bert_config, -bert_checkpoint, -bert_vocab
```
**运行`python keras_bert_ner/helper.py train --help`以获取参数细节。**

> **一些技巧**

如果你的预训练模型是**ALBERT，记得填写参数`-albert`。**

如果你想要**获取最佳的训练结果**，你需要**设置early-stopping和reduce-learning-rate系列的参数**(见训练参数)，**记得填写参数`-best_fit`。**

> **样例**

见`./examples/train_example`。运行`bash run_train.sh`进行训练。 

以下是**CNN**下游网络和**RNN**下游网络的训练模板：

**CNN**

```bash
PRETRAINED_LM_DIR="/home/liushaoweihua/pretrained_lm/albert_small_chinese"
DATA_DIR="../data"
OUTPUT_DIR="../models"

python run_train.py \
    -train_data ${DATA_DIR}/train.txt \
    -dev_data ${DATA_DIR}/dev.txt \
    -save_path ${OUTPUT_DIR} \
    -bert_config ${PRETRAINED_LM_DIR}/albert_config.json \
    -bert_checkpoint ${PRETRAINED_LM_DIR}/albert_model.ckpt \
    -bert_vocab ${PRETRAINED_LM_DIR}/vocab.txt \
    -albert \
    -do_eval \
    -device_map "0" \
    -tag_padding "X" \
    -best_fit \
    -max_epochs 256 \
    -early_stop_patience 5 \
    -reduce_lr_patience 3 \
    -reduce_lr_factor 0.5 \
    -batch_size 64 \
    -max_len 64 \
    -learning_rate 5e-6 \
    -model_type "cnn" \
    -cnn_filters 128 \
    -cnn_kernel_size 3 \
    -cnn_blocks 4 \
    -dropout_rate 0.0 \
    -learning_rate 5e-5
```

**RNN**

```bash
PRETRAINED_LM_DIR="/home/liushaoweihua/pretrained_lm/albert_small_chinese"
DATA_DIR="../data"
OUTPUT_DIR="../models"

python run_train.py \
    -train_data ${DATA_DIR}/train.txt \
    -dev_data ${DATA_DIR}/dev.txt \
    -save_path ${OUTPUT_DIR} \
    -bert_config ${PRETRAINED_LM_DIR}/albert_config.json \
    -bert_checkpoint ${PRETRAINED_LM_DIR}/albert_model.ckpt \
    -bert_vocab ${PRETRAINED_LM_DIR}/vocab.txt \
    -albert \
    -do_eval \
    -device_map "0" \
    -tag_padding "X" \
    -best_fit \
    -max_epochs 256 \
    -early_stop_patience 5 \
    -reduce_lr_patience 3 \
    -reduce_lr_factor 0.5 \
    -batch_size 64 \
    -max_len 64 \
    -learning_rate 5e-6 \
    -model_type "cnn" \
    -cell_type "lstm" \
    -rnn_units 128 \
    -rnn_num_hidden_layers 2 \
    -dropout_rate 0.0 \
    -learning_rate 5e-5
```

**字/句精度会在训练阶段进行展示，并保存在所指定的训练参数`save_path`路径下。** 

## 测试阶段

> **数据形式**

```json
[
    "时空周转公众注册，当天秒下时空周转是一款非常靠谱的小额现金快捷贷款平台。时空周转贷款申请到下款全过程都是在手机上完成的。扫一扫，立即申请时空周转app功能1、极速放款:自动审核、极速放款、实时到账;2、流程简单:在线填写资料，芝麻信用授权即可贷款;3、信息安全:数据库加>密技术、保护借款人隐私;4、随借随还:无论何时何地、借款轻松，还款便捷。时空周转app亮点1、闪电借款：纯线上自动化审核，快至30分钟到账!2、额度灵活：单期借款、现金分期，万元额度任你选。3、门槛超低：无门槛、无担保，有身份证即可借款。关注我们，更多口子信息问：有人在时空周>转借款过吗答：时空周转正常情况下2-3小时以内，也有特殊情况。问：时空周转贷款审核需要多少时间，时空周转贷款审核时间多长答：提前还款后，也还是能继续在时空周转借款的。问：时空周转没有还清还可以申请吗？答：时空周转！不需要太多的条件，借款也很快！问：时空周转真的像广告里面说的没有信用卡也能贷款吗答：用过，不是的，时空周转最大的好处就是可以节省分期的手续费问：时空周转好用吗时空周转，使用的人多不多啊答：时空周转它的手续费是比较低的，而且还款压力也比较小的。低于银行七哩八。5万块两年利息才6000多，分期手续费1200左右，算下来在时空周转年息只有7个点。还可以，一般吧，平台不错问：手机QQ里突然多了个时空周转，是什么？有谁知道？答：时空周转一般不会给借款人留的联系人打电话的，打电话主要是为了核实借款人提供的信息是否真实，再次确定借款人是否值得信赖。绝对不会透漏个人隐私",
    "抢红包、做任务、缺销路、缺人脉、推广产品，来这就对了!兼职赚钱+聚集人脉!本篇文章将对两种人群进行分析，一种是不做任何项目，只撸些APP拉新，做做任务赚钱的纯羊毛党，另一种是做项目，带团队的专职网赚者.通过本篇文章，你们会知道这才是你想要的东西.首先讲纯羊毛党:打开此>款APP，我发现，这里真是纯羊毛党的天堂，对于羊毛党，我只介绍这两个功能.1.红包大厅:有做项目的为了推广项目，会发些图方广告，以红包的形式，你只要观看10秒，就能抢到红包，对于羊毛党来说，向来是以量取胜的，而且你若有心，你会发现不少的优质项目.一手资源如图:2.任务大厅:就跟>其它比如蚂蚁，牛帮，众人帮等任务平台一样.这里可以发任务，接任务.价格也不错，最有突出的一个就是有朋友圈任务，简单转发到朋友圈，最低是1元/条，两三块/条的也有.每天光接这些任务，都够你一天几十的了.如图:3.这里要顺便说下我的微信群，大家都知道，每个任务平台收徒弟，徒弟完>成任务，师傅是有奖励的，之前我建立了众人帮，牛帮，余赚网，闲趣赚这几个平台的徒弟群，每天返70%的奖励给大家，得到了很好的效果.大家做任务都很积极，也得到了更多的奖励.现在我又单独建立了全民推的群，因为我发现全民推，在我开通了金牌站长+VIP会员后，我能得到徒弟30%的奖励，>所以，这个数字是非常可观的，为此，请做任务的兄弟们，一定要走我链接，加我微信，进群享高额分红，每天红包雨让你爽翻天.往下拉，加我微信进群.第二种人群:专职网赚带队干项目的为什么一定要来这里呢?请先听下我的故事:最近我手上有不少的好项目，却一直建立不起来一个更大的团队，思前思后，主要是没有找到可以引流的池子.一直到我有天看朋友圈发现有人在推广这个APP，下载之后才发现这里是真正的流量池.好了，我只说到这，干项目的都不是傻子，我只发点图给你们看，其它自己揣摩.第一张，请注意，数字:1.79亿，再看中间广告页的点击率，这上面是不断刷新的这一张，是加粉中心页面，可以把自己微信，群，小程序，公众号，甚至货源发上去综上所述，我觉得全民推这款APP是值得大家下载安装的.点击左下角阅读原文做任务的伙伴们记得加我微信进群享每日70%分红!有钱大家一起赚才是正确的操作方式.跟我干有钱赚点击阅读原文注册往期精彩回顾淘宝评价自动变现小而美的项目全国招收云闪付推广员:18元/单，一单一结手机端POS机-店小友，手机就是POS机，央行支付牌照，用友集团旗下，值得拥有软银支付是什么?有了店小友，我为什么还要推广软银支付?",
    "2019健康行业趋势，住家创业，稳赚不亏",
...
]
```

> **测试参数**

运行`python keras_bert_ner/helper.py test`或`python keras_bert_ner/helper.py predict`，快速浏览测试参数。

```bash
(nlp) liushaoweihua@ai-server-8:~/projects/Ner/tools/Keras-Bert-Ner$ python keras_bert_ner/helper.py test
usage: helper.py [-h] -test_data TEST_DATA -model_configs MODEL_CONFIGS
                 [-output_path OUTPUT_PATH] [-device_map DEVICE_MAP]
helper.py: error: the following arguments are required: -test_data, -model_configs
```
**运行`python keras_bert_ner/helper.py test --help`或`python keras_bert_ner/helper.py predict --help`以获取参数细节。**

> **样例**

见`./examples/test_example`。运行`bash run_test.sh`进行测试。


## 部署阶段

> **样例**

见`./examples/deploy_example`。运行`bash run_deploy.sh`部署API。 

运行`python query.py "时空周转公众注册，当天秒下时空周转是一款非常靠谱的小额现金快捷贷款平台。"`进行查询，以获取实体结果。

## 一些常用的中文预训练模型

> **BERT**
* [Google_bert](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)
* [HIT_bert_wwm_ext](https://storage.googleapis.com/chineseglue/pretrain_models/chinese_wwm_ext_L-12_H-768_A-12.zip)

> **ALBERT**
* [Google_albert_base](https://storage.googleapis.com/albert_models/albert_base_zh.tar.gz)
* [Google_albert_large](https://storage.googleapis.com/albert_models/albert_large_zh.tar.gz)
* [Google_albert_xlarge](https://storage.googleapis.com/albert_models/albert_xlarge_zh.tar.gz)
* [Google_albert_xxlarge](https://storage.googleapis.com/albert_models/albert_xxlarge_zh.tar.gz)
* [Xuliang_albert_xlarge](https://storage.googleapis.com/albert_zh/albert_xlarge_zh_177k.zip)
* [Xuliang_albert_large](https://storage.googleapis.com/albert_zh/albert_large_zh.zip)
* [Xuliang_albert_base](https://storage.googleapis.com/albert_zh/albert_base_zh.zip)
* [Xuliang_albert_base_ext](https://storage.googleapis.com/albert_zh/albert_base_zh_additional_36k_steps.zip)
* [Xuliang_albert_small](https://storage.googleapis.com/albert_zh/albert_small_zh_google.zip)
* [Xuliang_albert_tiny](https://storage.googleapis.com/albert_zh/albert_tiny_zh_google.zip)

> **Roberta**
* [roberta](https://storage.googleapis.com/chineseglue/pretrain_models/roeberta_zh_L-24_H-1024_A-16.zip)
* [roberta_wwm_ext](https://storage.googleapis.com/chineseglue/pretrain_models/chinese_roberta_wwm_ext_L-12_H-768_A-12.zip)
* [roberta_wwm_ext_large](https://storage.googleapis.com/chineseglue/pretrain_models/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16.zip)

## 参考
* 项目最初始的框架参考了macanv的工作：[BERT-BiLSTM-CRF-NER](https://github.com/macanv/BERT-BiLSTM-CRF-NER)。
* 项目最重要的部分参考了苏剑林的工作：[bert4keras](https://github.com/bojone/bert4keras)。
* ALBERT中文预训练模型系列[albert_zh](https://github.com/brightmart/albert_zh)让中文NER任务能够在相对较短的推理时间和相对较高的预测精度下进行线上部署。
* [BERT](https://github.com/google-research/bert), [ALBERT](https://github.com/google-research/albert), [RoBERTa](https://github.com/pytorch/fairseq/tree/master/examples/roberta)。

感谢以上作者和项目的贡献！
