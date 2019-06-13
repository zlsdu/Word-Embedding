Elmo英文和中文词向量训练
==========

Elmo与word2vec\fasttext\glove最大的区别在于Elmo得到的是动态词向量，即ELMo得到的词向量不是固定的，是根据句子上下文语境有变化，Elmo训练词向量目前提供以下两种实现方式

1、训练英文词向量
--------------
结合tensorflow_hub库来实现，下面先给出一个简单的示例，从简单的示例中讲解如何利用tensorflow_hub库加载elmo模型进行词向量训练，再给出一个完整elmo示例代码，elmo预训练词向量用于下游任务 

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/elmo1.png)

如上便是通过tensorflow_hub加载elmo模型对数据进行词向量预训练得到了output，我们先打印一下output看得到的是个什么东西

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/elmo2.png)

从上图可以看到elmo模型训练出来得到了一个字典，字典中key有`elmo`、`default`、`lstm_output1`、`lstm_outtput2`、`word_emb`、`sequence_len`，解释一个key的含义：<br>
(1) `sequence_len`: 输入中每个句子的长度<br>
(2) `word_emb`: elmo的最开始一层的基于character的word embedding，shape为[batch_size, max_length, 512]<br>
(3) `lstm_outpus1/2`: elmo中的第一层和第二层LSTM的隐状态输出，shape为[batch_size, max_length, 1024]<br>
(4) `default`: 前面得到的均为word级别的向量, default给出了使用mean-pooling求的句子级别的向量，即将上述elmo的所有词取平均<br>
(5) `elmo`: 每个词的输入层(word_emb)，第一层LSTM输出，第二层LSTM输出的线性加权之后的最终的词向量，此外这个线性权重是可训练的，shape为[batch_size, max_length, 1024]<br>
注意Attention，红色Attention：一般情况我们便是使用使用output['elmo']即可得到每个词的elmo词向量, 用于后续的任务<br>

可以打印查看一下我们刚才训练得的可以用于下游任务的动态词向量

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/elmo3.png)

ELMO预训练接下游任务使用详细参见源代码<br>

2、根据自己预料训练中文词向量模型
--------------------------
对官方发布的tensorflow的版本进行修改，训练自己的中文预料得到适用于中文词向量训练的elmo模型，大致步骤如下：<br>
(1) 准备好中文预料分词后的数据<br>
(2) 使用语料库生成词表数据<br>
(3) 预训练word2vec词向量<br>
(4) 下载github上bilm-tf代码[bilm-tf](https://github.com/allenai/bilm-tf) <br>
(5) 对bilm-tf代码进行修改训练中文词向量，下面将详细介绍如何操作<br>
本次试验用的是头条新闻类分类数据<br>

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/elmo4.png)

注意两点：<br>
* word2vec预训练词向量只需要根据上面提到词向量训练方式对语料库进行预训练即可<br>
* 词表数据vocab.data：使用语料库生成词表，但需要在词表开始额外添加如下开头三行

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/elmo5.png)

这一点原因是因为bilm-tf源码中加载词表时首三行需要读入这个信息，如下图所示

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/elmo6.png)

下载好bilm-tf源码后，下面我们着重介绍一下如何修改源码用于中文语料训练

(1) 修改bin/train_elmo.py文件<br>

此文件为程序入口，修改部分如下

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/elmo7.png)

解释<br>
* 将load_vocab函数的第二个参数修改为None<br>
* 设定运行elmo程序可见的GPU数<br>
* n_train_tokens：其大小间接影响程序迭代次数，可结合语料库大小进行设置<br>
* 将此部分完全注释掉<br>

(2) 修改bilm/train.py文件<br>

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/elmo8.png)

如上图首先修改加载预训练词向量信息，初始化Word Embeddings，并同时将参数trainable设置为True，其次另一处修改如下图所示，将最终得到的 
model.embedding_weights保存到hdf5文件中

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/elmo9.png)


(3) 在bilm-tf目录下，启动训练程序，启动命令如下<br>

```Bash
nohup python -u bin/train_elmo.py \
--train_prefix='/nfs/project/wangyun/data/toutiao_word_corpus.txt' \
--vocab_file /nfs/project/wangyun/data/vocab.data \
--save_dir /nfs/project/wangyun/bilm-tf/model >output 2>&1 &
```

参数train_prefix是训练语料库路径<br>
参数vocab_file是准备好的词表路径<br>
参数save_dir是运行中模型保存路径<br>
经过较长时间，运行结束后，产生模型文件如下<br>

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/elmo10.png)

(4) 在bilm-tf目录下，运行bin/dump_weights.py将checkpoint转换成hdf5文件<br>

```Bash
python -u  bin/dump_weights.py  \
--save_dir /nfs/project/wangyun/bilm-tf/model2  \
--outfile /nfs/project/wangyun/bilm-tf/model2/weights.hdf5
```

最终得到的模型文件如下：<br>

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/elmo11.png)

至此elmo中文模型已经训练结束，已经得到了vocab.data对应的vocab_embedding.hdf5文件，以及elmo模型对应的weights.hdf5文件和options.json文件，可以使用usage_token.py文件训练得到词向量<br>

(5) 修改usage_token.py文件，运行usage_token.py文件得到词向量<br>

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/elmo12.png)

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/elmo13.png)

python usage_token.py：便可得到词向量数据



Elmo相关原理简介
===========

1、ELMO模型结构
------------

ELMO首先根据名字Embedding from language model便可以ELMO是一个基于语言模型的词向量预训练模型，其次ELMO区别于word2vec、fasttext、glove静态词向量无法表示多义词，ELMO是动态词向量，不仅解决了多义词问题而且保证了在词性上相同<br>

ELMO模型使用语言模型Language Model进行训练，ELMO预训练后每个单词对应三个Embedding向量:<br>

(1) 底层对应的是Word Embedding，提取word的信息<br>
(2) 第一层双向LSTM对应是Syntactic Embedding，提取高于word的句法信息<br>
(3) 第二层双向LSTM对应的是Semantic Embedding，提取高于句法的语法信息<br>

ELMO在下游任务中是将每个单词对应的三个Embedding按照各自权重进行累加整合成一个作为新特征给下游任务使用，如下图所示：

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/elmo14.png)

在Bert论文中也给出了ELMO的模型图，比上图更简洁易于理解：

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/elmo15.png)

下面通过公式来再深入理解一下ELMO的双向LSTM语言模型，有一个前向和后向的语言模型构成，目标函数是取这两个方向语言模型的最大似然<br>

给定N个tokens，前向LSTM结构为：<br>

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/elmo16.png)

后向LSTM结构为：

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/elmo17.png)

Bi-LSTM的目标函数既是最大化前向和后向的对数似然和：

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/elmo18.png)


2、ELMO的优缺点
---------------

ELMO的优点便是使用了双层Bi-LSTM，并且最终模型学到的是Word Embedding + Syntactic Embedding + Semantic Embedding的线性组合<br>

ELMO相较于Bert模型来说，有以下缺点：<br>

(1) ELMO在特征抽取器选择方面使用的是LSTM，而不是更好用Transformer，Bert中使用的便是Transformer，Transformer是个叠加的自注意力机制构成的深度网络，是目前NLP里最强的特征提取器<br>
(2) ELMO采用双向拼接融合特征，相对于Bert一体化融合特征方式可能较弱<br>



