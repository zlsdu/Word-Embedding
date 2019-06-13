
word2vec中文词向量训练<br>
======

1、genism库<br>
-------
gensim库中提供了word2vec的cbow模型和skipgram模型的实现，可直接调用

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/word2vec1.png)

完整版参考代码<br>
<br>

2、tensorflow实现skipgram模型<br>
-------
skipgram模型使用中心词预测上下文，网上介绍很多了也可直接去看论文
本模型实验采用的数据时wiki百科中文数据，有原版和分词后版本，数据量大下载请移步
实现详细直接看代码，代码中关键处都有注释，这里提一下word2vec中常用的nce loss损失函数，nce loss函数个参数定义如下

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/word2vec2.png)

解释一下参数sampled_values，从tensorflow的nce_loss源代码中可以看到当sampled_ values=None时采样方式，word2vec中负采样过程其实就是优选采样词频高的词作负样本

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/word2vec3.png)

在上图中展现了nce_loss在实际使用过程中参数列表以及各个参数的含义，下面我们看一下tensorflow源码中对于nce_loss函数的实现逻辑：

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/word2vec4.png)

Tensorflow实现skipgram模型完整细节参考代码，训练测试效果可参见下图：

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/word2vec5.png)

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/word2vec6.png)


word2vec细节简介
===============

1、word2vec种语言模型
------------------
word2vec属于预测式词向量模型，两种Skipgram和CBOW<br>

(1) skipgram通过中间词预测周围词构建网络模型<br>

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/word2vec7.png)

(2) cbow通过周围词预测中间词构建网络模型<br>

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/word2vec8.png)


2、word2vec中Tricks:
------------------

(1) Hierarchical softmax<br>

分层softmax最主要的改进既是：不需要对网络W个单词的输出计算概率分布，只需要评估个节点即可<br>

(2) Negative sampling<br>

详细介绍一下负采样，word2vec中训练技巧：负采样<br>
通过模型的一次训练来解释负采样的过程，以skip_gram模型训练为例进行讲解<br>

训练样本输入词是：love，输出词是me，如果词典中有10000个词，设定训练出的词向量大小为300，则开始love和me都是通过one-hot编码的，在输出位置上me对应的是1，其他位置都是0，我们认为这些0位置对应的词即是负样本，1位置对应的是正样本，在不采用负采样情况下隐层到输出层的参数为300*10000，负采样的意思即是我们只在9999个负样本中选择很少一部分对应的参数进行更新（包括正样本的也更新），其他没有挑中的负样本参数保持不变，例如我们选择5个negative words进行参数更新，加上一个正样本总共是6个，因此参数是300*6，大大提高每次训练的计算效率，论文中指出对于小规模数据集我们选择5-20个negative words较好，在数据集情况下选择2-5个负样本较好<br>

(3) Subsampling of Frequent words<br>

频繁词的二次采样，根据论文描述在大的语料库中，频繁词如容易出现很多次的the\in\a提供的信息量远没有罕见词提供的信息量多，因此在后续的训练中频繁词无法提供更多的信息甚至会将网络带偏，因此提出了频繁词二次采样方式：即在每次训练时按照如下公式对训练集的单词wi进行丢弃：<br>

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/word2vec9.png)

CBOW模型的优化函数(skipgram模型类似)：<br>

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/word2vec10.png)

(4) word2vec是静态词向量预训练模型，词向量是固定的，不能解决多义词问题，无法考虑预料全局信息<br>
