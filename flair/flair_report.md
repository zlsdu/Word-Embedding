Flair Embedding预训练词向量介绍
======================

Flair Embedding来自Flair社区项目中的一个功能，Flair的功能包括<br>

(1) flair是一个强大的NLP库，允许将最先进的模型应用在文本上，模型包括：NER命名实体识别，PoS词性标注，语义消歧和分类等<br>

(2) flair支持多种语言，并且目前支持一个模型多个语言，即只用一个模型预测多种语言输入文本的NER任务和PoS任务<br>

最后flair也是一个word embedding库，目前支持的有Flair Embedding、Bert Embedding、Elmo Embedding、Glove Embedding、Fasttext Embedding等，同时flair库还支持组合各种Word Embedding，通过加载Flair社区提供的预训练模型便可得到word embeddings，下面通过实例看如何使用，预训练模型到时候会提供给大家<br>

首先我们看一下Flair Embedding的使用：

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/flair1.png)

使用十分方便，Bert、Elmo、Glove、Fasttext Embedding使用方式也都大同小异，接下来我们在看一下如何组合各种Embedding模型<br>

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/flair2.png)

Flair除了提供Word Embeddings外，还提供了Character Embeddings(以char作为特征)和Byte Pair Embeddings(即将单词切分成子序列，粒度要大于char)
注意：对于Flair预训练模型下载需要能够翻墙，否则模型无法下载模型, 所有示例完整代码请参见<br>



Flair原理简介
==========
Flair Embedding预训练目前听到的还不太多，当时有论文证明在NER任务上目前比BERT效果还要好，其他任务还不确定，下面是在NER任务上的对比<br>

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/flair3.png)

这里结合论文简要介绍一下Flair Embedding的预训练模型，并给出Flair Embedding源码github地址，上面详细介绍了Flair Embedding的使用<br>

1、Flair Embedding预训练模型
----------------------
A trained character language model to produce a novel type of word embeddin as contextual string embeddings<br>
(1) pre-train on large unlabeled corpora<br>
(2) capture word meaning in context and therefore produce different embeddings for polysemous words depending on their usage<br>
(3) model words and context fundamentally as sequences of characters, to both better handle rare and misspelled words as well as model subword structures such as prefixes and endings.<br>

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/flair4.png)

Character language model: 2048 * 1 layer<br>
1 Billion word corpus in 1 week for 1 GPU<br>
Sequence tagging model: 150 epochs<br>
256 * 1Layer<br>
Classic word embedding: <br>
GloVe , character feature: 25 * 1 layer<br>


