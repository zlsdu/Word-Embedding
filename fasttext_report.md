fasttext中文词向量训练
==========
fasttext原理好文请移步：
[fastText原理及实践](https://zhuanlan.zhihu.com/p/32965521)<br>


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
可调参数有：词向量的维度、subwords范围、epoch、learning_rate、thread<br>

```Bash
$ ./fasttext skipgram -input data/fil9 -output result/fil9 -minn 2 -maxn 5 
-dim 100 –epoch 2 –lr 0.5 –thread 4
```
<br>
同时可以用训练出来的词向量模型进行词向量打印和临近词向量查询等操作，自然在用命令进行训练前需要准备好数据，即准备好中文分词或是分字的文本数据作为数据
fasttext详细词向量训练过程以及各参数含义请移步博文：[fastText原理和文本分类实战，看这一篇就够了](https://blog.csdn.net/feilong_csdn/article/details/88655927)
此文中都有详细的介绍
