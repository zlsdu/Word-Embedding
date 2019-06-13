
word2vec说明
1、genism库
gensim库中提供了word2vec的cbow模型和skipgram模型的实现，可直接调用
![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/word2vec1.png)
完整版参考代码

2、tensorflow实现skipgram模型
skipgram模型使用中心词预测上下文，网上介绍很多了也可直接去看论文
本模型实验采用的数据时wiki百科中文数据，有原版和分词后版本，数据量大下载请移步
实现详细直接看代码，代码中关键处都有注释，这里提一下word2vec中常用的nce loss损失函数，nce loss函数个参数定义如下
![https://github.com/zlsdu/Word-Embedding/blob/master/phone/word2vec2.png]

解释一下参数sampled_values，从tensorflow的nce_loss源代码中可以看到当sampled_ values=None时采样方式，word2vec中负采样过程其实就是优选采样词频高的词作负样本
![https://github.com/zlsdu/Word-Embedding/blob/master/phone/word2vec3.png]

在上图中展现了nce_loss在实际使用过程中参数列表以及各个参数的含义，下面我们看一下tensorflow源码中对于nce_loss函数的实现逻辑：
![https://github.com/zlsdu/Word-Embedding/blob/master/phone/word2vec4.png]

Tensorflow实现skipgram模型完整细节参考代码，训练测试效果可参见下图：

![https://github.com/zlsdu/Word-Embedding/blob/master/phone/word2vec5.png]

![https://github.com/zlsdu/Word-Embedding/blob/master/phone/word2vec6.png]
