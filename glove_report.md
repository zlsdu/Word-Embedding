glove中文词向量训练
================

1、glove实战
----------

glove可通过官方发布的模型来训练中文词向量，步骤如下<br>

(1) 首先从官方下载glove并解压: [glove官网](https://github.com/stanfordnlp/GloVe)

(2) 将需要训练的预料处理好，做好分词去停用词处理，然后放到glove文件根目录下<br>

(3) 修改demo.sh文件，共有两处修改，如下图所示：

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/glove1.png)

(4) 在glove主文件下即根目录下执行命令进行编译： `make` <br>

(5) 在glove根目录下执行训练词向量命令： `bash demo.sh`  <br> 

如果训练数据大，训练时间长，使用如下命令训练词向量<br>

```Bash
nohup bash demo.sh > output.txt 2>&1 &
```

训练成功后，便会得到vectors.txt文件，可通过gensim的Word2Vec模块加载

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/glove2.png)

glove训练得到的词向量效果如下：

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/glove3.png)

gensim加载glove训练的词向量，glove和word2vec得到词向量文件区别在于word2vec包含向量的数量及其维度<br>

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/glove4.png)

可以使用gensim加载glove训练的词向量，因此先把glove格式词向量转化为word2vec的词向量，然后通过`gensim.models. KeyedVectors.load_word2vec_format()`加载glove训练的词向量模型


glove相关原理简介
================

1、Glove矩阵分解模型
-------------
Glove是一种矩阵分解式词向量预训练模型，如果我们要得到目标词w的预训练Embedding目标词w的Embedding表示取决于同语境中的词c的共现关系，因此引入矩阵分解的共现矩阵M，下面先给出共现矩阵M定义：<br>
* |Vw|行，每行对应Vw中出现的词w<br>
* |Vc|列，每列对应Vc中出现的词c<br>
* Mij表示wi和cj之间的某种关联程度，最简单的联系是w和c共同出现的次数<br>

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/glove5.png)

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/glove6.png)


(1) 重点来了，Glove中定义w和c的关联度为：<br>

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/glove7.png)


(2) Glove共现矩阵分解方式：<br>

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/glove8.png)


(3) Glove分解误差及优化目标定义：<br>

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/glove9.png)

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/glove10.png)


2、Glove中Tricks
-------------
(1) 优点：glove矩阵分解是基于全局预料的，而word2vec是基于部分预料训练的<br>

(2) 缺点：glove和word2vec、fasttext一样词向量都是静态的，无法解决多义词问题，另外通过glove损失函数，还会发现glove有一个致命问题，看下解释：<br>

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/glove11.png)

看上面的公式，glove损失函数显示，在glove得出的词向量上加上任意一个常数向量后，仍旧是损失函数的解，这问题就较大，如我们加上一个特别的常数，词向量就是十分接近了，这样就失去了词向量的表示含义，因此用glove训练出词向量后要进行check

