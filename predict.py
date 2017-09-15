# encoding: utf-8
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from model import Model
from load_data import get_vocab_dict, LoadTrainData, LoadTestData, get_word_vector
from test import test
from util import FLAGS

# 加载词典
vocab_dict = get_vocab_dict()
word_vectors = get_word_vector()
vocab_size = len(vocab_dict)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8

with tf.Session(config=config) as sess:
    # 此处需要根据model名字改
    log_dir = FLAGS.save_dir
    model_path = os.path.join(log_dir, "model.ckpt-5.meta")
    # 加载结构，模型参数和变量
    print("importing model...")
    saver = tf.train.import_meta_graph(model_path)

    saver.restore(sess, tf.train.latest_checkpoint(log_dir))
    #sess.run(tf.global_variables_initializer())

    graph = tf.get_default_graph()
    '''
    # 根据次数输出的变量名和操作名确定下边取值的名字
    all_vars = tf.trainable_variables()
    for v in all_vars:
        print(v.name)

    for op in sess.graph.get_operations():
        print(op.name)
    '''
    query = graph.get_tensor_by_name("Inputs/query:0")
    title = graph.get_tensor_by_name("Inputs/title:0")
    feature_local = graph.get_tensor_by_name("Inputs/feature_local:0")
    label = graph.get_tensor_by_name("Inputs/labels:0")
    #score = graph.get_tensor_by_name("squeeze:0")
    score = graph.get_tensor_by_name("predictions:0")
    testing_set = LoadTestData(vocab_dict, FLAGS.dev_set, query_len_threshold=FLAGS.query_len_threshold,
                           title_len_threshold=FLAGS.title_len_threshold, batch_size=FLAGS.batch_size)
    total_data = 0
    right = 0
    top = 0
    precision_bottom = 0
    recall_bottom = 0
    cnt = 0
    for batch_data in testing_set.next_batch():
        #batch_feature_local, queries, titles, labels, ranks = batch_data
        batch_feature_local, queries, titles, labels, index= batch_data
        fd = {query: queries,
              title: titles,
              feature_local: batch_feature_local,
              label: labels}

        res = sess.run(score, fd)
        #print("score:", res[:10])
        #print("label:", labels[:10])
        res = list(zip( index, labels, res.tolist()))
        df = pd.DataFrame(res, columns=[ 'index', 'label', 'score'])
        total_data += len(df)
        for i in range(0, len(df)):
            _label = df['label'][i]
            _score = df['score'][i]
            '''
            if _label >=1:
                recall_bottom += 1
            if _score >= 0.5:
                precision_bottom += 1
            if _label >=1 and _score >=  0.5:
                top += 1
            if (_label <1 and _score <  0.5) or (_label >= 1 and _score >=  0.5):
                right += 1
            '''
            if _label  == 2:
                recall_bottom += 1
            if _score == 2:
                precision_bottom += 1
            if _label ==2 and _score ==2:
                top += 1
            if _label == _score:
                right += 1
        df = df.drop('label', axis=1)
        if cnt ==0:
            df.to_csv("../predict/tencent.csv", mode='a', index=False)
            cnt += 1
        else:
            df.to_csv("../predict/tencent.csv", mode='a', index=False, header=False)

print("accuracy：", right / (total_data + 0.001))
print("precision:",top / (precision_bottom + 0.001))
print("recall:", top / (recall_bottom + 0.001))
print("=" * 60)