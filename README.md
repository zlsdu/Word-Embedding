# Word-Embedding
Word2vec，Fasttext，Glove，Elmo，Bert and Flair pre-train Word Embedding

本仓库详细介绍如何利用Word2vec，Fasttext，Glove，Elmo，Bert and Flair如何去训练Word Embedding，对算法进行简要分析，给出了训练详细教程以及源码，教程中也给出相应的实验效果截图<br>

1、环境
------------
* python>=3.5<br>
* tensorflow>=1.13<br>

2、Word Embedding教程快速链接
------------------
* [Word2vec中文词向量训练](https://github.com/zlsdu/Word-Embedding/blob/master/word2vec/word2vec_report.md)<br>
* [Fasttext中文词向量训练](https://github.com/zlsdu/Word-Embedding/blob/master/fasttext_report.md)<br>
* [Glove中文词向量训练](https://github.com/zlsdu/Word-Embedding/blob/master/glove_report.md)<br>
* [Elmo英文和中文词向量训练](https://github.com/zlsdu/Word-Embedding/blob/master/elmo/elmo_report.md)<br>
* [Bert句向量和词向量训练](https://github.com/zlsdu/Word-Embedding/blob/master/bert_report.md)<br>
* [Flair词向量训练](https://github.com/zlsdu/Word-Embedding/blob/master/flair/flair_report.md)<br>

3、实验数据简介
-------------------
* stop_words.txt: 停用词数据，用于数据预处理使用
* wiki.zh.text.jian: wiki百科简体中文原始数据，wiki.zh.text.jian.fenci: wiki百科简体中文分词后数据，wiki百科数据主要在word2vec的skipgram模型中使用，数据量较大，已放百度网盘，地址: [链接](https://pan.baidu.com/s/1DeIaIO35eWzZP75YRGNU9w), 密码: bvmw 
* toutiao_word_corpus.txt: 头条公开的新闻类分类数据，word2vec的gensim库实验、fasttext算法、glove算法中有使用


4、欢迎关注公众号
------------------

![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/gongzhonghao.png)


