
Bert句向量和词向量训练
===================

Bert训练句向量和词向量主要是利用了bert-as-service库进行训练，在服务器上安装bert-as-service环境，启动服务完成句向量和类似于elmo上下文相关的词向量训练，如下图所示：<br>

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/bert1.png)

注意由上述得到的是Word piece Embeddings而不是Word Embedding，因为使用Bert时，利用Bert模型Fine tuning效果远比使用Bert Embedding效果好，因此这里不对Bert Embedding做详细介绍，如果想要使用可以参考以下两个网址，里面有详细介绍：[bert-as-service](https://github.com/hanxiao/bert-as-service)、[bert-as-service详细文档](https://bert-as-service.readthedocs.io/en/latest/tutorial/token-embed.html)


Bert原理简介
============

1、Bert模型结构
------------

Bert相信NLPer都相当熟悉了，Bert模型主要两个特点<br>

(1) 特征提取器使用的是transformer<br>
(2) 预训练使用的是双向语言模型<br>

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/bert2.png)

Bert还有突出的地方在于它的预训练任务上，Bert采用了两个预训练任务：Masked语言模型(本质上是CBOW)、Next Sentence Prediction<br>

(1) Masked语言模型：随机选择语料中15%的单词，把它抠掉，也就是用[Mask]掩码代替原始单词，然后要求模型去正确预测被抠掉的单词，但15%词中只有80%会被替换成mask，10%被替换成另一个词，10%的词不发生改变<br>
(2) Next Sentence Prediction：指语言模型训练的时候，分两种情况选择句子，一种是正确拼接两个顺序句子，一种是从语料库中随机选择一个句子拼接到句子后面，做mask任务时顺带做句子关系预测，因此BERT的预训练是一个多任务过程在
Next Sentence Prediction可以让bert在跨句子的任务中表现的更好如句子相似度计算，而Masked LM则是让bert更好学到上下文信息<br>

因为Bert预训练预料丰富模型庞大，Bert的可适用的下游任务也很多，Bert可以对于上述四大任务改造下游任务，应用广泛：<br>
* 序列标注：分词、POS Tag、NER、语义标注<br>
* 分类任务：文本分类、情感计算<br>
* 句子关系判断：Entailment、QA、自然语言推断<br>
* 生成式任务：机器翻译、文本摘要等<br>

[Cross-lingual BERT：预训练任务Masked language Modeling、Translation language modeling，将很多种语言放到了一个词表中]<br>

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/bert3.png)

2、Bert中的细节理解
---------------
看下图，Bert在训练和使用过程中注意的一些小细节：<br>

* Bert训练的词向量不是完整的，而是WordPiece Embedding，因此要通过Bert模型得到英文Word Embedding要将WrodPiece Embeddings转化为Word Embedding<br>
* Bert预训练模型的输入向量是Token Embeddings + Segment Embeddings + Position Embeddings<br>
* 在对Bert模型微调进行下游任务时，需要知道Bert模型输出什么传入下游任务模型，即是开头[CLS]出的向量Special Classification Embeddings<br>

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/bert4.png)

3、特征提取插播
----------------
说到NLP中的特征提取器这里说一下，目前NLP常用的特征提取方式有CNN、RNN和Transformer，下面简要比较： <br>

(1)CNN的最大优点是易于做并行计算，所以速度快，但是在捕获NLP的序列关系尤其是长距离特征方面天然有缺陷<br>
(2)RNN一直受困于其并行计算能力，这是因为它本身结构的序列性依赖导致的<br>
(3)Transformer同时具备并行性好，又适合捕获长距离特征<br>

这里顺便放上ELMO、GPT、BERT的对比图，其中ELMO特征提取器使用的是RNN，GPT和Bert使用的是Transformer，GPT使用的是单向语言模型，ELMO和BERT使用的是双向语言模型<br>

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/bert5.png)

4、Attention机制插播
-------------------
为了更好理解Transformer这里希望可以通俗简短的介绍一下Attention机制<br>

(1)从机器翻译(Encoder-Decoder)角度看一下Attention机制(下面图片引自网络)<br>

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/bert6.png)

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/bert7.png)

先看上面第一张图，是传统的机器翻译，Y1由[X1,X2,X3,X4]编码得到，可以看出[X1,X2,X3,X4]对翻译得到词Y1贡献度没有区别<br>

再看第二张图是Attention + 机器翻译，每个输出的词Y受输入X1,X2,X3,X4影响的权重不同，这个权重便是由Attention计算，因此可以把Attention机制看成注意力分配系数，计算输入每一项对输出权重影响大小<br>

(2)从一个机器翻译实例理解Attention机制，了解一下Attention如何对权重进行计算(图片引自网络)<br>

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/bert8.png)


首先由原始数据经过矩阵变化得到Q、K、V向量，如下图

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/bert9.png)

以单词Thinking为例，先用Thinking的q向量和所有词的k向量相乘，使用下面公式：

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/bert10.png)

这种得到每个单词对单词Thinking的贡献权重，然后再用得到的权重和每个单词的向量v相乘，得到最终Thinking向量的输出

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/bert11.png)

还有Self-attention和Multi-head attention都是在Attention上进行一些改动，这样不详细介绍
