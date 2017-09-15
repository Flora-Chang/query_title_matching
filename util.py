# encoding: utf-8
import tensorflow as tf


flags = tf.app.flags

# Model parameters
flags.DEFINE_integer("filter_size", 64, "the num of filters of CNN")
flags.DEFINE_integer("embedding_dim", 100, "words embedding size")
flags.DEFINE_float("keep_prob", 0.8, "dropout keep prob")

# change each runing
flags.DEFINE_string("flag", "multi.tencent", "word/char/drmm")
flags.DEFINE_string("save_dir", "../logs/multi.tencent_lr0.0005_bz64_filter64_embedding100_1504755401", "save dir")
flags.DEFINE_string("predict_dir", "predict_word_1.csv", "predict result dir")


# Training / test parameters
flags.DEFINE_integer("query_len_threshold", 20, "threshold value of query length")
flags.DEFINE_integer("title_len_threshold", 20, "threshold value of document length")
flags.DEFINE_integer("batch_size", 64, "batch size")
flags.DEFINE_integer("num_epochs", 5, "number of epochs")
flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
flags.DEFINE_integer("pooling_size", 80, "pooling size")
flags.DEFINE_float("validation_steps", 61, "steps between validations")
flags.DEFINE_float("GPU_rate", 0.9, "steps between validations")

flags.DEFINE_string("training_set", "../data/train_tencent/train.txt", "training set path")
flags.DEFINE_string("train_set", "../data/train_tencent/train.test.txt", "train set path")
flags.DEFINE_string("dev_set", "../data/train_tencent/dev.txt", "dev set path")
flags.DEFINE_string("vocab_path", "../data/train_tencent/word_dict.txt", "vocab path")
flags.DEFINE_string("vectors_path", "../data/train_tencent/vectors_word.txt", "vectors path")

FLAGS = flags.FLAGS

