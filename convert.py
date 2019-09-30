#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:huanghui
# datetime:2019/9/30 10:12
from utils.ckpt2pb import convert

"""
convert调用参数说明
    :param task: 任务名，divorce，labor， loan
    :param tagDir: 任务标签文件，tags.txt
    :param originDir: 若是文件夹，则选择最后一个模型，若是文件名，则选择该模型文件。
    :param convertDir: 生成的pb模型名叫model.pb，在该目录下
    :param model_type: bert后接的模型类型，rcnnatt，如果为None直接接全连接层
    :param bert_dir: bert预训练模型文件夹，下边只需要包含配置文件和词典
    """
convert(task="divorce", tagDir="data/divorce/tags.txt", originDir="ckpt/divorce",
        convertDir="pb/divorce", model_type="rcnnatt", bert_dir="/home/huanghui/data/chinese_L-12_H-768_A-12")
