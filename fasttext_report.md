fasttext中文词向量训练
==========
fasttext原理好文请移步：
[fastText原理和文本分类实战](https://blog.csdn.net/feilong_csdn/article/details/88655927)<br>


1、fasttext词向量实战简介
-------------
facebook在github上发布了fasttext用于文本分类和词向量训练的官方代码，可直接用于中文词向量训练，下载网址是：[fasttext](https://github.com/facebookresearch/fastText)，下载下来之后首先需要make编译一下，编译通过之后便可直接使用如下命令进行中文词向量训练，也可对参数进行调节：<br>

```Bash
$ ./fasttext skipgram -input data/fil9 -output result/fil9
```
<br>

上面命令是使用skipgram进行词向量学习，当然也可以使用cbow进行词向量学习<br>


```Bash
$ ./fasttext cbow -input data/fil9 -output result/fil9
```
<br>

可调参数有：`词向量的维度`，`subwords范围`， `epoch`， `learning_rate`， `thread` <br>

```Bash
$ ./fasttext skipgram -input data/fil9 -output result/fil9 -minn 2 -maxn 5 
-dim 100 –epoch 2 –lr 0.5 –thread 4
```
<br>

同时可以用训练出来的词向量模型进行词向量打印和临近词向量查询等操作，自然在用命令进行训练前需要准备好数据，即准备好中文分词或是分字的文本数据作为数据
fasttext详细词向量训练过程以及各参数含义请移步博文，此文中都有详细的介绍
[fastText原理和文本分类实战](https://blog.csdn.net/feilong_csdn/article/details/88655927)


Flair Embedding细节简单介绍
===================

fastText是一个快速文本分类算法，与基于神经网络的分类算法相比有两大优点：<br>

(1) fastText在保持高精度的情况下加快了训练速度和测试速度<br>
(2) fastText不需要预训练好的词向量，fastText会自己训练词向量<br>

因此fasttext在预训练上的体现便是我们可以通过fasttext训练自己预料的词向量<br>

1、fasttext模型架构
---------------
fastText模型架构和word2vec中的CBOW很相似， 不同之处是fastText预测类别标签而CBOW预测的是中间词，即模型架构类似但是模型的任务不同，下面是fasttext的结构图<br>

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/fasttext1.png)

2、fasttext中tricks
------------------

fasttext和word2vec类似还体现在优化tricks上，fasttext的tricks有：<br>

(1) Hierarchical softmax：减少计算量<br>
(2) n-gram特征：n-gram好处有可以考虑到语序信息同时也可避免OOV问题<br>

fasttext接下来在文本分类专题中会详细介绍，这里需明白fasttext能够快速训练词向量提供nlp任务的预训练Embedding，且实验证明fasttext是预训练中的佼佼者


