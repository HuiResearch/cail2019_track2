#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:huanghui
# datetime:2019/9/30 9:42

import tensorflow as tf
from bert import tokenization
from tensorflow.python.platform import gfile
import numpy as np
from tqdm import tqdm

pre_batch_size = 100

class InputFeatures(object):
  """A single set of features of data."""
  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
      """
      Initialize the input segment.

      Args:
          self: (todo): write your description
          input_ids: (str): write your description
          input_mask: (todo): write your description
          segment_ids: (str): write your description
          label_id: (str): write your description
          is_real_example: (bool): write your description
      """
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example

def convert_single_example(sent, label_list, max_seq_length,
                           tokenizer):
    """
    Converts a list of sentences into a list of sentences.

    Args:
        sent: (todo): write your description
        label_list: (list): write your description
        max_seq_length: (int): write your description
        tokenizer: (todo): write your description
    """
  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = sent

  if len(tokens_a) > max_seq_length - 2:
    tokens_a = tokens_a[0:(max_seq_length - 2)]
    # bound = int((max_seq_length-2)/2)
    # tokens_a = tokens_a[0:bound] + tokens_a[(len(tokens_a)-bound):]
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  input_mask = [1] * len(input_ids)

  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
##label是一个列表
  # label_id = label_map[example.label]

  label_id = [0]*len(label_map)
  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature

class BERTModel:

    def __init__(self, task, pb_model, tagDir, threshold, vocab_file):
        """
        :param task: 任务类型，包括divorce，labor，loan
        :param pb_model: pb模型文件名
        :param tagDir: 任务标签tags文件
        :param threshold: 多标签分类的阈值列表
        :param vocab_file: bert词典文件 vocab.txt
        """
        tf.reset_default_graph()
        domain2len = {"divorce": 128, "labor": 150, "loan": 200}
        self.max_seg_length = domain2len[task]
        self.pb_model = pb_model
        self.vocab_file = vocab_file
        self.label_dir = tagDir

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpu_config)
        ##加载阈值列表
        self.threshold = threshold

        f = open(self.label_dir, 'r', encoding='utf-8')
        lines = f.readlines()
        self.label = []
        for line in lines:
            self.label.append(line.strip())
        f.close()
        # 生成label和id转换字典
        self.label2id = {}
        for (i, label) in enumerate(self.label):
            self.label2id[label] = i
        self.id2label = {value: key for key, value in self.label2id.items()}
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=self.vocab_file, do_lower_case=True)
        self.sess = tf.Session()
        with gfile.FastGFile(pb_model, 'rb') as f:
            self.graph = tf.GraphDef()
            self.graph.ParseFromString(f.read())
            self.sess.graph.as_default()
            tf.import_graph_def(self.graph, name='')
        self.sess.run(tf.global_variables_initializer())
        self.input_ids_p = self.sess.graph.get_tensor_by_name('input_ids:0')
        self.input_mask_p = self.sess.graph.get_tensor_by_name('input_mask:0')
        self.segment_ids_p = self.sess.graph.get_tensor_by_name('segment_ids:0')
        self.probabilities = self.sess.graph.get_tensor_by_name('pred_prob:0')

    def convert(self, line):
        """
        Converts a text of a list of a document.

        Args:
            self: (todo): write your description
            line: (todo): write your description
        """
        feature = convert_single_example(line, self.label, self.max_seg_length, self.tokenizer)
        input_ids = feature.input_ids
        input_mask = feature.input_mask
        segment_ids = feature.segment_ids
        label_ids = feature.label_id
        return input_ids, input_mask, segment_ids, label_ids

    def getAllResult(self, sentences):
        """一次性预测所有句子"""
        step = int(len(sentences) / pre_batch_size)
        all_result = []
        for i in tqdm(range(step)):
            result = self.predict(sentences[i * pre_batch_size:(i + 1) * pre_batch_size])
            all_result.extend(result)
        if len(all_result) < len(sentences):
            result = self.predict(sentences[len(all_result):])
            all_result.extend(result)
        return all_result

    def rematch(self, arrays):
        """
        Removes indices of the indices.

        Args:
            self: (todo): write your description
            arrays: (array): write your description
        """
        predict_list = []
        for array in arrays:
            temp = []
            for i in range(len(array)):
                if array[i] == 1:
                    temp.append(self.id2label[i])
            predict_list.append(temp)
        return predict_list

    """predict返回的是一个二维列表，存储预测结果[[], ['DV1', 'DV2']]"""
    def predict(self, sentences):
        """预测小批量句子"""
        def getPre(arr, id2label):
            """
            Get the label for each label.

            Args:
                arr: (array): write your description
                id2label: (str): write your description
            """
            predict_list = []
            for i in range(len(arr)):
                if arr[i] > self.threshold[i]:
                    predict_list.append(id2label[i])
            return predict_list

        def getPredictLabel(array, id2label):
            """
            Return the label for a label.

            Args:
                array: (array): write your description
                id2label: (str): write your description
            """
            proba = array[0]
            result = []
            for p in proba:
                result.append(getPre(p, id2label))

            return result

        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []

        for sentence in sentences:
            sentence = self.tokenizer.tokenize(sentence)
            input_ids, input_mask, segment_ids, label_ids = self.convert(sentence)
            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)

        feed_dict = {self.input_ids_p: input_ids_list,
                     self.input_mask_p: input_mask_list,
                     self.segment_ids_p: segment_ids_list}
        probabilities_ = self.sess.run([self.probabilities], feed_dict)

        result = getPredictLabel(probabilities_, self.id2label)
        return result

    def getProb(self, sentences):
        """
        Returns a list of sentences

        Args:
            self: (todo): write your description
            sentences: (todo): write your description
        """
        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []

        for sentence in sentences:
            sentence = self.tokenizer.tokenize(sentence)
            input_ids, input_mask, segment_ids, label_ids = self.convert(sentence)
            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)

        feed_dict = {self.input_ids_p: input_ids_list,
                     self.input_mask_p: input_mask_list,
                     self.segment_ids_p: segment_ids_list}
        probabilities_ = self.sess.run([self.probabilities], feed_dict)

        return probabilities_[0]

    def getProbs(self, sentences):
        """
        Parameters ---------- sentences : list of sentences

        Args:
            self: (todo): write your description
            sentences: (todo): write your description
        """
        step = int(len(sentences) / pre_batch_size)
        all_result = []
        for i in tqdm(range(step)):
            probs = self.getProb(sentences[i * pre_batch_size:(i + 1) * pre_batch_size])
            all_result.extend(probs)
        if len(all_result) < len(sentences):
            probs = self.getProb(sentences[len(all_result):])
            all_result.extend(probs)
        all_result = np.asarray(all_result)
        return all_result