#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:huanghui
# datetime:2019/9/30 9:58

from bert import modeling
import os
from tensorflow.python.framework import graph_util
from utils.models import *

modelMap = {"rcnnatt": RCNNATT, "rcnn": RCNN}

def create_model(bert_config, input_ids, input_mask, segment_ids,
                 num_labels, model_type=None):
    """
    :param bert_config:
    :param input_ids:
    :param input_mask:
    :param segment_ids:
    :param num_labels: 类别数
    :param model_type: bert后接的模型类型，rcnn，rcnnatt
    :return: sigmoid后的结果
    """
    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)

    if model_type:
        embedding = model.get_sequence_output()
        model_layer = modelMap[model_type](
            embedding=embedding, context_dim=200, hidden_dim=200, dropout_keep_prob=1.0
        )
        output_layer = model_layer.getLogits()
    else:
        output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        output_layer = tf.nn.dropout(output_layer, keep_prob=1.0)
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

    probabilities = tf.nn.sigmoid(logits)
    return probabilities


def convert(task, tagDir, originDir, convertDir, model_type, bert_dir):
    """
    :param task: 任务名，divorce，labor， loan
    :param tagDir: 任务标签文件，tags.txt
    :param originDir: 若是文件夹，则选择最后一个模型，若是文件名，则选择该模型文件。
    :param convertDir: 生成的pb模型名叫model.pb，在该目录下
    :param model_type: bert后接的模型类型，rcnn，orgin,模型类型都为小写
    :param bert_dir: bert预训练模型文件夹，下边只需要包含配置文件和词典
    """
    tf.reset_default_graph()
    domain2len = {"divorce": 128, "labor": 150, "loan": 200}
    max_seg_length = domain2len[task]

    if not os.path.exists(convertDir):
        os.makedirs(convertDir)
    f = open(tagDir, 'r', encoding='utf-8')
    lines = f.readlines()
    label = []
    for line in lines:
        label.append(line.strip())
    f.close()
    num_labels = len(label)

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)
    graph = tf.get_default_graph()
    with graph.as_default():
        input_ids_p = tf.placeholder(tf.int32, [None, max_seg_length], name="input_ids")
        input_mask_p = tf.placeholder(tf.int32, [None, max_seg_length], name="input_mask")
        segment_ids_p = tf.placeholder(tf.int32, [None, max_seg_length], name="segment_ids")
        bert_config = modeling.BertConfig.from_json_file(os.path.join(bert_dir, 'bert_config.json'))
        probabilities = create_model(
            bert_config=bert_config, input_ids=input_ids_p, input_mask=input_mask_p,
            segment_ids=segment_ids_p, num_labels=num_labels, model_type=model_type
        )
        probabilities = tf.identity(probabilities, 'pred_prob')
        saver = tf.train.Saver()
        if os.path.isdir(originDir):
            saver.restore(sess, tf.train.latest_checkpoint(originDir))
        else:
            saver.restore(sess, originDir)
        tmp_g = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['pred_prob'])
    with tf.gfile.GFile(os.path.join(convertDir, "model.pb"), 'wb') as f:
        f.write(tmp_g.SerializeToString())

