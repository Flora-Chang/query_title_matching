# encoding: utf-8
import numpy as np
import json
from util import FLAGS

def get_vocab_dict(input_file=FLAGS.vocab_path):
    words_dict = {}
    cnt = 0
    with open(input_file) as f:
        for word in f:
            words_dict[word.strip()] = cnt
            cnt += 1
    return words_dict

def get_word_vector(input_file=FLAGS.vectors_path):
    word_vectors = []
    with open(input_file) as f:
        for line in f:
            line = [float(v) for v in line.strip().split()]
            word_vectors.append(line)
    return word_vectors

# output batch_major data
def batch(inputs, threshold_length):
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    '''
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
        max_sequence_length = min(max_sequence_length, threshold_length)
    '''
    max_sequence_length = threshold_length
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)  # == PAD
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            if j >= threshold_length:
                sequence_lengths[i] = max_sequence_length
                break
            inputs_batch_major[i, j] = element
    # [batch_size, max_time] -> [max_time, batch_size]
    # inputs_time_major = inputs_batch_major.swapaxes(0, 1)
    return inputs_batch_major

class LoadTrainData(object):
    def __init__(self, vocab_dict, data_path, query_len_threshold, title_len_threshold, batch_size=64):
        self.vocab_dict = vocab_dict
        self.batch_size = batch_size
        self.title_len_threshold = title_len_threshold  # 句子长度限制
        self.query_len_threshold = query_len_threshold
        self.data = open(data_path, 'r').readlines()
        self.batch_index = 0
        print("len data: ", len(self.data))

    def _word_2_id(self, word):
        if word in self.vocab_dict.keys():
            res = self.vocab_dict[word]
        else:
            res = self.vocab_dict['UNK']
        return res

    def next_batch(self, shuffle=True):
        self.batch_index = 0
        self.cnt = 0
        data = np.array(self.data)
        data_size = len(data)
        num_batches_per_epoch = int(data_size / self.batch_size) + 1
        print("training_set: ", data_size, num_batches_per_epoch)

        if shuffle:
            np.random.shuffle(data)
        shuffled_data = data

        while self.batch_index < num_batches_per_epoch \
                and (self.batch_index + 1) * self.batch_size <= data_size:
            queries = []
            titles = []
            labels = []
            ranks = []
            batch_feature_local = []
            start_index = self.batch_index * self.batch_size
            self.batch_index += 1
            end_index = min(self.batch_index * self.batch_size, data_size)
            batch_data = shuffled_data[start_index:end_index]

            for line in batch_data.tolist():
                line = line.strip().split('\t')
                self.cnt +=1
                label = int(line[0])
                """
                if label == 2:
                    label = 1
                else:
                    label = 0
                """
                ori_query = line[2].split()
                query = list(map(self._word_2_id, ori_query))
                ori_title = line[1].split()
                title = list(map(self._word_2_id, ori_title))
                rank = int(line[3].strip())
                queries.append(query)
                titles.append(title)
                labels.append(label)
                ranks.append(rank)
                local_match = np.zeros(shape=[self.query_len_threshold, self.title_len_threshold], dtype=np.int32)
                for i in range(min(self.query_len_threshold, len(ori_query))):
                    for j in range(min(self.title_len_threshold, len(title))):
                        if ori_query[i] == ori_title[j]:
                            local_match[i, j] = 1
                batch_feature_local.append(local_match)  # [batch_size, query_length, title_length]

            queries = batch(queries, self.query_len_threshold)
            titles = batch(titles, self.title_len_threshold)
            yield batch_feature_local, queries, titles, labels,ranks
        print("self.cnt:", self.cnt)


class LoadTestData(object):
    def __init__(self, vocab_dict, data_path, query_len_threshold, title_len_threshold, batch_size=64):
        self.vocab_dict = vocab_dict
        self.query_len_threshold = query_len_threshold
        self.title_len_threshold = title_len_threshold
        self.index = 0
        self.data = open(data_path, 'r').readlines()
        self.data_size = len(self.data)
        self.batch_size = batch_size
        self.cnt = 0

    def _word_2_id(self, word):
        if word in self.vocab_dict.keys():
            res = self.vocab_dict[word]
        else:
            res = self.vocab_dict['UNK']
        return res

    def next_batch(self):
        while (self.index ) * self.batch_size < self.data_size:
            if (self.index + 1) * self.batch_size <= self.data_size:
                batch_data = self.data[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            else:
                batch_data = self.data[self.index * self.batch_size: self.data_size]
            self.index += 1
            queries = []
            titles = []
            labels = []
            ranks = []
            batch_feature_local = []
            for line in batch_data:
                self.cnt += 1
                line = line.strip().split('\t')
                label = int(line[0])
                """
                if label == 2:
                    label = 1
                else:
                    label = 0
                """
                ori_query = line[2].split()
                query = list(map(self._word_2_id, ori_query))
                ori_title = line[1].split()
                title = list(map(self._word_2_id, ori_title))
                rank = int(line[3].strip())
                queries.append(query)
                titles.append(title)
                labels.append(label)
                ranks.append(rank)
                local_match = np.zeros(shape=[self.query_len_threshold, self.title_len_threshold], dtype=np.int32)
                for i in range(min(self.query_len_threshold, len(ori_query))):
                    for j in range(min(self.title_len_threshold, len(ori_title))):
                        if ori_query[i] == ori_title[j]:
                            local_match[i, j] = 1
                batch_feature_local.append(local_match)  # [batch_size, query_length, title_length]

            queries = batch(queries, self.query_len_threshold)
            titles = batch(titles, self.title_len_threshold)
            yield batch_feature_local, queries, titles, labels, ranks

        print("self.cnt:", self.cnt)


