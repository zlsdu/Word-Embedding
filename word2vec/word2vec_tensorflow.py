import jieba
import math
import random
import collections
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


class LoadData():
    '''
    This class is created by zhanglei at 2019/06/10.
    The environment: python3.5 or later and tensorflow1.10 or later.
    The functions include load data，build vocabulary，pretrain，generate batches.
    '''
    def __init__(self):
        '''
        数据加载相关参数设置
        '''
        self.corpus_path = '../../data/wiki.zh.text.jian.part'   #wiki中文语料库文件路径
        self.stop_words_path = '../../data/stop_words.txt'       #停用词文件路径
        self.vocabulary_size = 200000  #词典大小
        self.batch_size = 128          #batch大小
        self.num_skips = 2             #中心词使用的次数
        self.skip_window = 1           #skipgram算法窗口大小
        
    def read_data(self):
        """
        读取文本，把文本的内容的所有词放在一个列表
        self.stop_words_path:停用词文件路径
        self.corpus_path:语料库文件路径
        Returen:
            vocabularys_list = [词表]
        """
        # 读取经过预处理后的语料库数据
        lines = open(self.stop_words_path, 'r').readlines()
        stop_words = {word.strip():1 for word in lines}
        vocabularys_list = []
        wiki_zh_data = []
        with open(self.corpus_path, "r") as f:
            line = f.readline()
            while line:
                raw_words = list(jieba.cut(line.strip()))
                raw_words = [raw_words[i] for i in range(len(raw_words))
                             if raw_words[i] not in stop_words and raw_words[i] != ' ']
                vocabularys_list.extend(raw_words)
                wiki_zh_data.append(raw_words)
                line = f.readline()
        return vocabularys_list, wiki_zh_data
    
    def know_data(self, vocabularys, topk):
        '''
        查看语料库信息，包括词频数、最高词频词语
        Args:
            vocabularys: 所有经过分词去停用词后词语信息
        '''
        vocab_dict = {}
        for i in range(len(vocabularys)):
            if vocabularys[i] not in vocab_dict:
                vocab_dict[vocabularys[i]] = 0
            vocab_dict[vocabularys[i]] += 1
        vocab_list = sorted(vocab_dict.items(), key=lambda x:x[1], reverse=True)
        print('词典中词语总数:{}'.format(len(vocab_dict)))
        print('top{}词频词语信息:{}'.format(topk, vocab_list[:topk]))
        
    def build_dataset(self, vocabularys):
        '''
        对词表中词出现的个数进行统计,并且将出现的罕见的词设置成了 UNK,主要便是对需要处理的数据集进行了
        Args:
            vocabularys: 词典
        Return:
            data_index_list: [[word,index], [], ...]
            word_index_dict：{word:index, ...}
            index_word_dict: {index:word, ...}
        '''
        word_count_list = [['UNK', -1]]
        word_index_dict = {}
        index_word_dict = {}
        data_index_list = []
        #extend是扩展添加list中的内容
        #collections.Counter是将数字按key统计称dict的形式,并且按照了数量进行了排序, most_common方法是将格式转化成list，并且只取参数的数量
        word_count_list.extend(collections.Counter(vocabularys).most_common(self.vocabulary_size - 1))
        for word, _ in word_count_list:
            word_index_dict[word] = len(word_index_dict)
        unk_count = 0
        for word in vocabularys:
            if word in word_index_dict:
                index = word_index_dict[word]
            else:
                index = 0  
                unk_count += 1
            data_index_list.append(index)
        word_count_list[0][1] = unk_count
        index_word_dict = dict(zip(word_index_dict.values(), word_index_dict.keys()))
        return data_index_list, word_index_dict, index_word_dict, word_count_list
    
    def generate_batch(self):
        '''
        其中collections.deque(maxlen)是python中的双向列表，通过设置maxlen则列表会自动控制大小
        当append新的元素到尾部时，便会自动将头部的元素删除，始终保持 2*skip_window+1的窗口大小
        batch_size: batch大小
        num_skips: 中心词使用的次数，中心词预测窗口中label重复使用的次数
        skip_window: 设置窗口大小，窗口大小为2*skip_window+1
        Return:
            batch_text: 一个batch文本
            batch_label: 一个batch标签
        '''
        global data_index
        assert self.batch_size % self.num_skips == 0    #断言：判断assert condition中condition是真是假
        assert self.num_skips <= 2 * self.skip_window
        batch_text = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        batch_label = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        span = 2 * self.skip_window + 1  # [ skip_window target skip_window ]
        # buffer中存储的一个窗口2*skip_window的总共数据
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(data_index_list[data_index])
            data_index = (data_index + 1) % len(data_index_list)
        # num_skips代表一个中心词使用的次数，因为便需要控制num_skips和skip_window的大小关系
        # 通过skip_window拿到处于中间位置的词，然后用他去预测他周围的词,周围词选取是随机的
        for i in range(self.batch_size // self.num_skips):
            target = self.skip_window  # target label at the center of the buffer
            targets_to_avoid = [self.skip_window]
            for j in range(self.num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch_text[i * self.num_skips + j] = buffer[self.skip_window]
                batch_label[i * self.num_skips + j, 0] = buffer[target]
            # 在一个中心词num_skips次数结束之后，便将窗口往后移动，重新加一个新的词进来
            # maxlen会自动保持窗口的总大小，都会自动往回移动一个单词
            buffer.append(data_index_list[data_index])
            data_index = (data_index + 1) % len(data_index_list)
        return batch_text, batch_label


class SkipgramModel():
    '''
    This class is created by zhanglei at 2019/06/10.
    The environment: python3.5 or later and tensorflow1.10 or later.
    The functions include set parameters，build skipgram model.
    '''
    def __init__(self, valid_examples):
        '''
        skipgram模型相关参数设置，并加载模型
        '''
        self.batch_size = 128           #batch大小
        self.vocabulary_size = 200000   #词典大小
        self.embedding_size = 256       #word embedding大小
        self.num_sampled = 32         #负采样样本的数量
        self.learning_rate = 0.5        #学习率
        self.valid_examples = valid_examples
        self.skipgram()
        
    def skipgram(self):
        '''
        skipgram模型结构
        '''
        tf.reset_default_graph()
        self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
        valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)
        #word embedding进行随机初始化
        with tf.name_scope('initial'):
            self.embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
            self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            nce_weights = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                            stddev=1.0 / math.sqrt(self.embedding_size)))
            nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]),dtype=tf.float32)

        with tf.name_scope('loss'):
            #采用nce_loss损失函数，并进行负采样
            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                             biases=nce_biases, 
                                             inputs=self.embed, 
                                             labels=self.train_labels,
                                             num_sampled=self.num_sampled, 
                                             num_classes=self.vocabulary_size))
        #使用梯度下降优化算法
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        
        #验证集，筛选与验证集词向量相似度高的词向量
        with tf.name_scope('valid'):
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
            self.normalized_embeddings = self.embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings, valid_dataset)
            self.similarity = tf.matmul(valid_embeddings, self.normalized_embeddings, transpose_b=True)


def train_model(loadData, skipgram):
    '''
    启动tf.Session()加载模型进行训练
    '''
    num_steps = 100000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = loadData.generate_batch()
            feed_dict = {skipgram.train_inputs: batch_inputs, skipgram.train_labels: batch_labels}
            _, loss_val = sess.run([skipgram.optimizer, skipgram.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 100 == 0:
                if step > 0:
                    average_loss /= 2000
                print("Average loss at step {} : {}".format(step, average_loss))
                average_loss = 0

            if step % 1000 == 0:
                sim = skipgram.similarity.eval()
                for i in range(len(skipgram.valid_examples)):
                    valid_word = index_word_dict[skipgram.valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[:top_k]
                    log_str = "Nearest to %s:" % valid_word
                    for k in range(top_k):
                        close_word = index_word_dict[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)
                    print(log_str)
        final_embeddings = skipgram.normalized_embeddings.eval() 


if __name__ == '__main__':
    #加载数据类
    loadData = LoadData()
    #读取语料库原文件，得到词表信息
    vocabularys, wiki_zh_data = loadData.read_data()
    loadData.know_data(vocabularys, topk=10)
    data_index_list, word_index_dict, index_word_dict, word_count_list = loadData.build_dataset(vocabularys)
    valid_word = ['中国', '学院', '中心', '北京', '大学', '爱', "不错", "中文", "幸福"]  #验证集
    valid_examples =[word_index_dict[li] for li in valid_word]    #验证机index
    global data_index
    data_index = 0
    #加载skipgram模型
    skipgram = SkipgramModel(valid_examples)
    #进行模型训练，最终得到word2vec模型副产物word embedding
    train_model(loadData, skipgram)

