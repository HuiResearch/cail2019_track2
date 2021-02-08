# cail2019_track2
中国法研杯CAIL2019要素抽取任务第三名方案分享
====
欢迎大家使用[tensorflow1.x的bert系列模型库，支持单机多卡，梯度累积，自动导出pb部署](https://github.com/huanghuidmml/textToy)

（修改了一下readme，之前那一版感觉写的太水了。）

这次比赛和前两名差距很大，但是也在此给大家分享一下我所用的方案。

主要的trick包括领域预训练、focal loss、阈值移动、规则匹配以及模型优化、调参。

没有使用模型融合。

###  **效果对比**

由于是第一次参赛，很多比赛细节没有做记录，效果对比的分数是我从凭印象在上传历史记录里边找的，可能分数不一致，但是大概就在那个范围，还请见谅。

| Model | 详情 | 线上评分 |
| :------: | :------: | :------: |
| BERT | 使用bert_base做多标签分类 | 69.553 |
| BERT+RCNN+ATT | 在BERT后增加RCNN层，并把最大池化换成Attention | 70.143 |
| BERT+RCNN+ATT | 增加阈值移动 | 70.809 |
| BERT+RCNN+ATT | 增加focal loss | 71.126 |
| BERT+RCNN+ATT | 增加规则 | 72.2 |
| BERT+RCNN+ATT | 使用比赛数据预训练BERT | 72.526 |
| BERT+RCNN+ATT | copy样本正例少的数据（divorce loan有效） | 72.909 |
| BERT+RCNN+ATT | 在比赛数据基础上增加裁判文书（1000篇）做预训练（labor有效） | 73.483 |
| BERT+RCNN+ATT | 增加否定词规则 | 73.533 |

### **主要参数**

| 参数名 | 参数值 |
| :------: | :------: |
| 预训练模型 | [BERT_Base_Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) |
| max_length(divorce) | 128 |
| max_length(labor) | 150 |
| max_length(loan) | 200 |
| batch_size | 32 |
| learning_rate | 2e-5 |
| num_train_epochs | 30 |
| alpha(focal loss) | 0.25 |
| gamma(focal loss) | 2 |
| hidden_dim(lstm) | 200 |

**方案介绍**
------
### **任务简介**
根据给定司法文书中的相关段落，识别相应的关键案情要素，其中每个句子对应的类别标签个数不定，属于多标签问题。任务共涉及三个领域，包括婚姻家庭、劳动争议、借款合同。
例如：

| 例句 | 标签 |
| :------: | :------: |
| 高雷红提出诉讼请求：1、判令被告支付原告的工资1630×0.8×4＝5216元； | ["LB2"] |
| 原告范合诉称：原告系被告处职工。 | [] |
| 5、判令被告某甲公司支付2011年9月8日至2012年8月7日未签订劳动合同的二倍工资差额16，298.37元； | ["LB9", "LB6"] |

根据数据集，我选定的方案就是传统的多标签分类方法。bert预训练模型使用的是google开源的bert_base_chinese.

### **任务难点**

* **正负例样本不均衡**
* **有的要素标签正例仅有几条，模型无法学习**

### **解决方案**

#### **focal loss**
减少易分类样本的权重，增加难分类样本的损失贡献值，参数见上表的alpha，gamma

#### **阈值移动**
将比赛的数据集切分为训练集和测试集。先用训练集去训练模型，
然后使用测试集去测试模型，筛选阈值；最后把所有数据拿去训练最后的提交模型，
预测阈值就采用之前筛选出来的阈值。

#### **copy少量数据**
数据增强我尝试过eda，但是效果不行，不如不用，后来使用copy的方法做数据增强，
将正例少的样本copy一定的数量，但是不能copy太多，否则会严重破坏分布。
而且这个方法我只在divorce和loan两种领域有提升，labor上下降了，
可能是copy量不合理，大家可以下去尝试修改一下，看下会不会提升。

#### **模型优化**
最后使用的模型是BERT + RCNN，并且RCNN部分的最大池化修改为Attention。
主要方法就是将BERT的输出向量X输入BiLstm，得到一个特征向量H，最后将X和H
拼接送入Attention。

#### **规则**
规则主要是为了修正模型无法学习的要素标签，使用的方式：首先通过
标签的解释说明和包含标签的样本确定规则，规则在python中使用的是正则
表达式；然后针对需要预测的文本，我们先使用正则表达式去匹配，若是
匹配成功，则说明文本包含该规则对应的标签；最后把规则匹配出来的标签与
模型预测的标签取并集，得到最终预测要素集。

规则举例：
> ['.(保证合同|抵押合同|借款合同).(无效|不发生效力).*']
   ，对应的要素是LN12。
 
**否定词规则**

否定词规则的意思是：在采用规则修正的时候，若是句子以一些否定词结尾，规则将不生效。

举例：

> 被告五金公司辩称本案借款合同和保证合同均无效，缺乏法律依据，本院**不予采纳**。

> 实际标签: LN13 LN10

这个句子可以匹配到我们写的LN12的规则：‘.*(保证合同|抵押合同|借款合同).*(无效|不发生效力).*‘

但是因为末尾出现了不予采纳，所以该标签规则不生效，没有LN12。

#### **领域预训练**

bert模型采用的是bert_base_chinese，如果使用徐亮大佬的roberta应该还会有提升。

司法领域属于特殊领域，所以使用比赛数据先做了一次预训练，在三种领域都有一定的提升，
后边我爬取一些裁判文书来做预训练，可能是因为数据量小和质量不够，只在labor上得到了
提升，如果保证数据量和质量，应该会有提升。

**代码说明**
-------

#### **基本代码**
CUDA_VISIBLE_DEVICES=1是指定第一块显卡，根据具体情况自己改，
如果CPU的话就不用了。

**训练**

> CUDA_VISIBLE_DEVICES=1 python train.py

**将ckpt转为pb**

> CUDA_VISIBLE_DEVICES=1 python convert.py

**线下测试**

> CUDA_VISIBLE_DEVICES=1 python evaluation.py

#### **如果需要额外预训练的话，使用以下代码**

**创建预训练数据txt**

> python genPretrainData.py

**创建预训练数据的tfrecord文件**

> python createPretrainData.py

**预训练**

> CUDA_VISIBLE_DEVICES=1 python run_pretrain.py

**Reference**
-----
1. [TensorFlow code and pre-trained models for BERT](https://github.com/google-research/bert)
2. [The implementation of focal loss proposed on "Focal Loss for Dense Object Detection" by KM He and support for multi-label dataset.](https://github.com/ailias/Focal-Loss-implement-on-Tensorflow)

**感谢**
-----
感谢队友牧笛的帮助
