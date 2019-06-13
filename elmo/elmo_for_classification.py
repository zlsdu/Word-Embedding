import os
import re
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub


class LoadData():
    '''
    The functions include load data
    '''
    def __init__(self):
        '''Initialize kinds of parameters，adjust the parameters accroding to data'''
        self.seq_length = 128
        
    def load_directory_data(self, directory):
        '''Load all files from a directory in a DataFrame.'''
        data = {}
        data["sentence"] = []
        data["sentiment"] = []
        for file_path in os.listdir(directory):
            with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
                data["sentence"].append(f.read())
                data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
        return pd.DataFrame.from_dict(data)

    def load_dataset(self, directory):
        '''Merge positive and negative examples, add a polarity column and shuffle'''
        pos_df = self.load_directory_data(os.path.join(directory, "pos"))
        neg_df = self.load_directory_data(os.path.join(directory, "neg"))
        pos_df["polarity"] = 1
        neg_df["polarity"] = 0
        return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

    def download_and_load_datasets(self, force_download=False):
        '''加载斯坦福提供的英文文本分类训练数据'''
        dataset = tf.keras.utils.get_file(
            fname="aclImdb.tar.gz", 
            origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", 
            extract=True)

        train_df = self.load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "train"))
        test_df = self.load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "test"))
        return train_df, test_df

    def get_train_test_data(self, train_df, test_df):
        '''
        将数据有DataFrame转化为可用的np.array形式
        Args:
            train_df:训练数据DataFrame
            test_df:测试数据DataFrame
        Return:
            train_data = [train_text, train_label, train_label_one_hot]
            test_data = [test_text, test_label, test_label_one_hot]
        '''
        # Create datasets (Only take up to 150 words for memory)
        train_text = train_df['sentence'].tolist()
        train_text = [' '.join(t.split()[0:self.seq_length]) for t in train_text]
        train_text = np.array(train_text, dtype=object)[:, np.newaxis]
        train_label = train_df['polarity'].tolist()
        train_label_one_hot = [[1, 0] if train_label[i]==0 else [0, 1] for i in range(len(train_label))]

        test_text = test_df['sentence'].tolist()
        test_text = [' '.join(t.split()[0:self.seq_length]) for t in test_text]
        test_text = np.array(test_text, dtype=object)[:, np.newaxis]
        test_label = test_df['polarity'].tolist()
        test_label_one_hot = [[1, 0] if test_label[i]==0 else [0, 1] for i in range(len(test_label))]
        train_data = [train_text, train_label, train_label_one_hot]
        test_data = [test_text, test_label, test_label_one_hot]
        return train_data, test_data


class ElmoModel_for_Classification():
    '''
    This class is created by zhanglei at 2019/06/10.
    The environment: python3.5 or later and tensorflow1.10 or later.
    The functions include set parameters，build elmo model + two lstm for classification.
    '''
    def __init__(self):
        '''
        Initialize kinds of parameters，adjust the parameters accroding to data
        '''
        self.batch_size = 32          #训练batch大小
        self.seq_length = 128         #序列长度
        self.embedding_size = 1024    #embedding大小，elmo模型是固定的1024
        self.hidden_size = 128        #隐层神经元个数
        self.class_num = 2            #分类类别数，二分类
        self.keep_prob = 0.5          #防止过拟合
        self.num_layers = 2           #lstm层数
        self.learning_rate = 1e-3     #学习率大小
        self.elmo_model()

    def elmo_model(self):
        '''
        build model
        '''
        tf.reset_default_graph()
        self.input_x = tf.placeholder(tf.string, [None, 1], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None, 2], name='input_y')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=True)
        self.embedding = self.elmo(tf.squeeze(tf.cast(self.input_x, tf.string), axis=1), as_dict=True, signature='default')
        
        # LstmCell单元的隐层数取决于上一层embedding_size的大小
        with tf.name_scope('rnn'):
            cell = tf.contrib.rnn.BasicLSTMCell(self.embedding_size, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            cells = [cell for _ in range(self.num_layers)]
            rnn_cells = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            # 下面不使用dtype还会报错，需要初始化
            outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cells, inputs=self.embedding['elmo'], dtype=tf.float32)
            last_outputs = outputs[:, -1, :]
        
        with tf.name_scope('hidden'):
            fc = tf.layers.dense(last_outputs, self.hidden_size, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
        
        with tf.name_scope('logits'):
            # tf.math.argmax
            self.logits = tf.layers.dense(fc, self.class_num, name='fc2')
            self.y_pred_cls = tf.arg_max(tf.nn.softmax(self.logits), 1)
        
        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            # tf.nn.softmax_cross_entropy_with_logits_v2
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            self.optim = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step)  


def train_model(train_data, test_data, loadData, elmoModel):
    '''
    启动图进行模型训练
    Args:
        train_data:训练数据np.array
        test_data:测试数据np.array
        loadData:数据加载类对象
        elmoModel:模型加载类对象
    Return
    '''
    train_text, train_label, train_label_one_hot = train_data
    test_text, test_label, test_label_one_hot = test_data
    data_size = len(train_label_one_hot)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_batch_every_epoch = math.ceil(data_size/elmoModel.batch_size)
        y_pred = np.zeros(shape=data_size, dtype=np.int32)
        print(num_batch_every_epoch)
        #开始迭代训练
        for i in range(num_batch_every_epoch):
            start_idx, end_idx = elmoModel.batch_size*i, elmoModel.batch_size*(i+1)
            if end_idx > data_size:
                end_idx = data_size
                act_end_id = elmoModel.batch_size - end_idx + data_size
                batch_y_pred = batch_y_pred[:act_end_id]
            batch_train_text = train_text[start_idx:end_idx]
            batch_train_label = train_label_one_hot[start_idx:end_idx]
            train_loss, _, batch_y_pred = sess.run([elmoModel.loss, elmoModel.optim, elmoModel.y_pred_cls],
                                feed_dict={elmoModel.input_x:batch_train_text, elmoModel.input_y:batch_train_label})
            print(train_loss)
            y_pred[start_idx:end_idx] = batch_y_pred 


if __name__ == '__main__':
    loadData = LoadData()
    train_df, test_df = loadData.download_and_load_datasets()
    train_data, test_data = loadData.get_train_test_data(train_df, test_df)
    elmoModel = ElmoModel_for_Classification()
    train_model(train_data, test_data, loadData, elmoModel)	




