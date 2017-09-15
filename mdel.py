# encoding: utf-8
import tensorflow as tf
from util import FLAGS

class Model(object):
    def __init__(self, max_query_word, max_title_word, word_vec_initializer, batch_size, filter_size,
                 vocab_size, embedding_size, learning_rate, keep_prob):
        self.word_vec_initializer = word_vec_initializer
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.max_query_word = max_query_word
        self.max_title_word = max_title_word
        self.filter_size = filter_size
        self.local_output = None
        self.distrib_output = None
        self._input_layer()
        self.optimizer(self.feature_local, self.query, self.title)
        self.merged_summary_op = tf.summary.merge([self.sm_loss_op])

    def _input_layer(self):
        with tf.variable_scope('Inputs'):
            self.feature_local = tf.placeholder(dtype=tf.float32,
                                                shape=(None, self.max_query_word, self.max_title_word),
                                                name='feature_local')
            self.query = tf.placeholder(dtype=tf.int32, shape=(None, self.max_query_word), name='query')
            self.title = tf.placeholder(dtype=tf.int32, shape=(None, self.max_title_word), name='title')
            self.labels = tf.placeholder(dtype=tf.int32, shape=[None,], name='labels')

    def _embed_layer(self, query, title):
        with tf.variable_scope('Embedding_layer'), tf.device("/cpu:0"):
            self.embedding_matrix = tf.get_variable(name='embedding_matrix',
                                                    initializer=self.word_vec_initializer,
                                                    dtype=tf.float32,
                                                    trainable=False)
            self.sm_emx_op = tf.summary.histogram('EmbeddingMatrix', self.embedding_matrix)
            embedding_query = tf.nn.embedding_lookup(self.embedding_matrix, query)
            embedding_title = tf.nn.embedding_lookup(self.embedding_matrix, title)
            return embedding_query, embedding_title

    def local_model(self, feature_local, is_training=True, reuse=False):
        with tf.variable_scope('local_model'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            features_local = tf.reshape(feature_local, [-1, self.max_query_word, self.max_title_word])
            conv = tf.layers.conv1d(inputs=features_local, filters=self.filter_size, kernel_size=[1],
                                    activation=tf.nn.tanh) #[?,max_query_word,1,self.filter_size]
            conv = tf.reshape(conv, [-1,self.filter_size*self.max_query_word]) #[?,max_query_word*self.filter_size]
            dense1 = tf.layers.dense(inputs=conv, units=self.filter_size, activation=tf.nn.tanh) #[?, self.filter_size]
            #dropout = tf.layers.dropout(inputs=dense1, rate=self.keep_prob, training=is_training) #extra add
            dense2 = tf.layers.dense(inputs=dense1, units=self.filter_size, activation=tf.nn.tanh)
            self.local_output = dense2
            return self.local_output

    def distrib_model(self, query, title, is_training=True, reuse=False):
        with tf.variable_scope('Distrib_model'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            embedding_query, embedding_title = self._embed_layer(query=query, title=title)
            with tf.variable_scope('distrib_query'):
                query = tf.reshape(embedding_query,
                                   [-1, self.max_query_word, self.embedding_size, 1])  # [?, max_query_word(=15), self.embedding_size,1]
                conv1 = tf.layers.conv2d(inputs=query, filters=self.filter_size,
                                         kernel_size=[3, self.embedding_size],
                                         activation=tf.nn.tanh, name="conv_query")  # [?,15-3+1,1, self.filter_size]
                pooling_size = self.max_query_word - 3 + 1
                pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                                pool_size=[pooling_size, 1],
                                                strides=[1, 1], name="pooling_query")  # [?, 1,1 self.filter_size]
                pool1 = tf.reshape(pool1, [-1, self.filter_size])  # [?, self.filter_size]
                dense1 = tf.layers.dense(inputs=pool1, units=self.filter_size, activation=tf.nn.tanh, name="fc_query")
                self.distrib_query = dense1  # [?, self.filter_size]

            with tf.variable_scope('distrib_title'):
                title = tf.reshape(embedding_title, [-1, self.max_title_word, self.embedding_size, 1])
                conv1 = tf.layers.conv2d(inputs=title, filters=self.filter_size,
                                         kernel_size=[3, self.embedding_size],
                                         activation=tf.nn.tanh, name="conv_title")  # [?,15-3+1,1, self.filter_size]
                pooling_size = self.max_title_word - 3 + 1
                pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                                pool_size=[pooling_size, 1],
                                                strides=[1, 1], name="pooling_title")  # [?, 1,1 self.filter_size]
                pool1 = tf.reshape(pool1, [-1, self.filter_size])  # [?, self.filter_size]
                dense1 = tf.layers.dense(inputs=pool1, units=self.filter_size, activation=tf.nn.tanh, name="fc_title")
                self.distrib_title = dense1  # [?, self.filter_size]


            distrib = tf.multiply(self.distrib_query, self.distrib_title) #[?, self.filter_size]
            distrib = tf.reshape(distrib,[-1,self.filter_size]) #[?, self.dims2]
            fuly1 = tf.layers.dense(inputs=distrib, units=self.filter_size, activation=tf.nn.tanh)
            #drop = tf.layers.dropout(inputs=fuly1, rate=self.keep_prob, training=is_training)  # extra add
            fuly2 = tf.layers.dense(inputs=fuly1, units=self.filter_size, activation=tf.nn.tanh)
            self.distrib_output = fuly2
            print("distrib_output:",self.distrib_output)
            return self.distrib_output

    def ensemble_model(self, feature_local, query, title, is_training=True, reuse=False):
        with tf.variable_scope('emsemble_model'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            self.model_output = tf.concat([self.local_model(is_training=is_training, feature_local=feature_local, \
                                                            reuse=reuse),self.distrib_model(is_training=is_training, \
                                                            query=query,title=title,reuse=reuse)], axis=-1)
            fuly = tf.layers.dense(inputs=self.model_output, units=self.filter_size, activation=tf.nn.tanh)
            fuly1 = tf.layers.dense(inputs=fuly, units=3, activation=tf.nn.sigmoid)

        output = fuly1
        return output

    def optimizer(self, feature_local, query, title):
        self.score = self.ensemble_model(feature_local=feature_local, query=query,
                                             title=title, is_training=True, reuse=False)  # [batch_size, 1]
        #self.score = tf.squeeze(self.score, -1, name="squeeze")  # [batch_size]
        print("score:",self.score)
        self.predictions = tf.argmax(self.score, 1, name="predictions")
        labels = tf.one_hot(self.labels, depth=3)
        self.losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=self.score)
        self.loss = tf.reduce_mean(self.losses)
        #self.loss = tf.reduce_mean(tf.square(self.score - self.labels))

        self.sm_loss_op = tf.summary.scalar('Loss', self.loss)

        with tf.name_scope("optimizer"):
            self.optimize_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.90, beta2=0.999,epsilon=1e-08).minimize(self.loss)

