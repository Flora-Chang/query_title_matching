# encoding: utf-8
#!/usr/bin/env python
import os
import time
import numpy as np
import tensorflow as tf
from util import FLAGS
from model import Model
from load_data import get_vocab_dict, LoadTrainData, LoadTestData, get_word_vector
from test import test

vocab_dict = get_vocab_dict()
word_vectors = get_word_vector()
vocab_size = len(vocab_dict)
#print("vocab_size: ",vocab_size)
#print("word_vector: ", len(word_vectors))

training_set = LoadTrainData(vocab_dict,
                             data_path=FLAGS.training_set,
                             query_len_threshold=FLAGS.query_len_threshold,
                             title_len_threshold=FLAGS.title_len_threshold,
                             batch_size=FLAGS.batch_size)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = FLAGS.GPU_rate

with tf.Session(config=config) as sess:
    timestamp = str(int(time.time()))
    print("timestamp: ", timestamp)
    model_name = "{}_lr{}_bz{}_filter{}_embedding{}_{}".format(FLAGS.flag,
                                            FLAGS.learning_rate,
                                            FLAGS.batch_size,
                                            FLAGS.filter_size,
                                            FLAGS.embedding_dim,
                                            timestamp)

    model = Model(max_query_word=FLAGS.query_len_threshold,
                  max_title_word=FLAGS.title_len_threshold,
                  word_vec_initializer=word_vectors,
                  batch_size=FLAGS.batch_size,
                  vocab_size=vocab_size,
                  embedding_size=FLAGS.embedding_dim,
                  learning_rate=FLAGS.learning_rate,
                  filter_size=FLAGS.filter_size,
                  keep_prob=FLAGS.keep_prob)

    log_dir = "../logs/" + model_name
    model_path = os.path.join(log_dir, "model.ckpt")
    os.mkdir(log_dir)
    init = tf.global_variables_initializer()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    steps = []
    step = 0
    num_epochs = FLAGS.num_epochs
    for epoch in range(num_epochs):
        print("epoch: ", epoch)
        for batch_data in training_set.next_batch():
            feature_local, queries, titles, labels,_ = batch_data
            feed_dict = {model.feature_local: feature_local,
                         model.query: queries,
                         model.title: titles,
                         model.labels: labels}
            _, loss, predictions, summary =\
                sess.run([model.optimize_op, model.loss, model.predictions, model.merged_summary_op],
                         feed_dict)

            if step % FLAGS.validation_steps == 0:
                #print("label", labels[0:10])
                #print("score", predictions[0:10])
                #print(step, " - loss:", loss)
                '''
                train_set = LoadTestData(vocab_dict, FLAGS.train_set,
                                         query_len_threshold=FLAGS.query_len_threshold,
                                         title_len_threshold=FLAGS.title_len_threshold, batch_size=FLAGS.batch_size)
                print("On training set:\n")
                test(sess, model, train_set, filename=None)
                dev_set = LoadTestData(vocab_dict, FLAGS.dev_set, query_len_threshold=FLAGS.query_len_threshold,
                                       title_len_threshold=FLAGS.title_len_threshold, batch_size=FLAGS.batch_size)
                print("On validation set:\n")
                test(sess, model, dev_set, filename=None)
            '''

        step += 1
        saver = tf.train.Saver(tf.global_variables())
        saver_path = saver.save(sess, os.path.join(log_dir, "model.ckpt"), step)
        train_set = LoadTestData(vocab_dict, FLAGS.train_set,
                                 query_len_threshold=FLAGS.query_len_threshold,
                                 title_len_threshold=FLAGS.title_len_threshold, batch_size=FLAGS.batch_size)
        print("On training set:\n")
        test(sess, model, train_set, filename=None)
        dev_set = LoadTestData(vocab_dict, FLAGS.dev_set, query_len_threshold=FLAGS.query_len_threshold,
                               title_len_threshold=FLAGS.title_len_threshold, batch_size=FLAGS.batch_size)
        print("On validation set:\n")
        test(sess, model, dev_set, filename=None)


    coord.request_stop()
    coord.join(threads)
