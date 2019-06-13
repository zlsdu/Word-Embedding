glove中文词向量训练
================

glove可通过官方发布的模型来训练中文词向量，步骤如下<br>
1、首先从官方下载glove并解压<br>

[glove官网](https://github.com/stanfordnlp/GloVe)
<br>

2、将需要训练的预料处理好，做好分词去停用词处理，然后放到glove文件根目录下<br>
3、修改demo.sh文件，共有两处修改，如下图所示：
![image](https://github.com/zlsdu/Word-Embedding/blob/master/phone/glove1.png)

4、在glove主文件下即根目录下执行命令进行编译： `make` <br>
5、在glove根目录下执行训练词向量命令： `bash demo.sh`  <br> 
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
