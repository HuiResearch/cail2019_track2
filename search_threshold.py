#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:huanghui
# datetime:2019/9/30 10:55

from utils.predict import BERTModel
import json
import os
from utils.evaluate import evaluate
import logging

logging.basicConfig(level=logging.INFO)

def getLab(probs, id2label, threshold):
    predict_list = []
    for i in range(len(probs)):
        if probs[i] > threshold[i]:
            predict_list.append(id2label[i])
    return predict_list

def getPreLab(array, id2label, threshold):
    result = []
    for p in array:
        result.append(getLab(p, id2label, threshold))
    return result

def load_file(filename):
    f = open(filename, "r", encoding='utf-8')
    all_sentence = []
    all_label = []
    for line in f:
        pre_doc = json.loads(line)
        for sent in pre_doc:
            all_sentence.append(sent["sentence"])
            all_label.append(sent["labels"])
    f.close()
    return all_sentence, all_label

def searchThreshold(domain, model_pb, threshold_dir,
                    test_file, tag_file, vocab_file):
    """
    用划分好的测试集取搜索最优的阈值，精度0.1,再低会过拟合，最好使用交叉验证来做
    由于交叉验证bert代价很大，就没做
    :param domain: 数据集类别，divorce、labor、loan
    :param model_pb: pb模型文件
    :param threshold_dir: 阈值搜索结果json文件存放地址
    :param test_file: 用来搜索阈值的测试文件
    :param tag_file: 标签tags文件
    :param vocab_file: bert模型词典文件
    :return: 将搜索的阈值存入threshold_dir，命名为threshold.json
                将搜索过程记录在search.json
    """
    thresholds = []
    for i in range(1, 10):
        thresholds.append(round(i * 0.1, 1))

    all_sentences, all_labels = load_file(test_file)

    logging.info("———— 开始加载模型 ————\n")
    model = BERTModel(task=domain, pb_model=model_pb, tagDir=tag_file, threshold=None, vocab_file=vocab_file)
    logging.info("———— 模型加载结束 ————\n")
    logging.info("———— 开始生成预测概率metric ————\n")
    probas = model.getProbs(all_sentences)
    logging.info("———— 预测概率metric生成结束 ————\n")

    result = {}
    result["domain"] = domain
    result["label_score"] = []
    logging.info("———— 开始搜索 %s 的最优阈值 ————\n" % domain)
    best_threshold = [0.5] * 20
    threshold_init = [0.5] * 20
    for i in range(20):
        best_score = 0
        label_result = {}
        scoreOfthreshold = {}
        label_result["label"] = i
        for j in range(len(best_threshold)):
            threshold_init[j] = best_threshold[j]
        ##遍历一开始初始化的候选阈值列表，0.1--0.9的九个候选阈值
        for threshold in thresholds:
            threshold_init[i] = threshold
            predicts = getPreLab(probas, model.id2label, threshold_init)
            score, f1 = evaluate(predict_labels=predicts, target_labels=all_labels, tag_dir=tag_file)
            scoreOfthreshold[threshold] = score
            if score > best_score:
                best_threshold[i] = threshold
                best_score = score
        label_result["score"] = scoreOfthreshold
        result["label_score"].append(label_result)
        logging.info(best_threshold)
        logging.info(label_result)
        logging.info("\n")
    result["best_threshold"] = best_threshold
    logging.info("搜索出来的阈值： %s \n" % best_threshold)
    logging.info("————开始将结果写入文件————\n")
    if not os.path.exists(threshold_dir):
        os.makedirs(threshold_dir)
    threshold_file = os.path.join(threshold_dir, "threshold.json")
    search_file = os.path.join(threshold_dir, "search.json")

    ouf_t = open(threshold_file, "w", encoding="utf-8")
    ouf_s = open(search_file, "w", encoding="utf-8")
    json.dump(best_threshold, ouf_t, ensure_ascii=False)
    json.dump(result, ouf_s, ensure_ascii=False)
    ouf_s.close()
    ouf_t.close()

if __name__ == '__main__':
    task = "divorce"

    """整理代码测试，测试文件test_file就用的训练文件，正式使用需要改为切分的测试数据集"""

    searchThreshold(domain=task, model_pb="pb/model.pb", threshold_dir="threshold",
                    test_file="data/divorce/train_selected.json", tag_file="data/divorce/tags.txt",
                    vocab_file="/home/huanghui/data/chinese_L-12_H-768_A-12/vocab.txt")
