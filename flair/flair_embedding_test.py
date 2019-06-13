from flair.data import Sentence
from flair.embeddings import WordEmbeddings
from flair.embeddings import CharacterEmbeddings
from flair.embeddings import BytePairEmbeddings
from flair.embeddings import BertEmbeddings, ELMoEmbeddings
from flair.embeddings import FlairEmbeddings, StackedEmbeddings

# 创建Sentense对象，Flair中共两个对象Sentense、token，sentense是由一系列token组成
sentence = Sentence('The grass is green .')
print(sentence)
print(sentence.get_token(4))
print(sentence[3])

# Glove Embeddings加载训练
glove_embedding_forward = WordEmbeddings('model/glove.gensim')
sentence = Sentence('The grass is green .')
glove_embedding_forward.embed(sentence)
for token in sentence:
    print(token)
    print(token.embedding)

#Fasttest Embedding加载训练
fasttext_embedding_forward = WordEmbeddings('model/zh-wiki-fasttext-300d-1M')
sentence = Sentence('The grass is green .')
fasttext_embedding_forward.embed(sentence)
for token in sentence:
    print(token)
    print(token.embedding)

#Flair Embedding加载训练
flair_embedding_forward = FlairEmbeddings('model/news-forward-0.4.1.pt')
sentence = Sentence('The grass is green .')
flair_embedding_forward.embed(sentence)
for token in sentence:
    print(token)
    print(token.embedding)

#Bert Embedding加载训练
embedding = BertEmbeddings()
sentence = Sentence('The grass is green .')
embedding.embed(sentence)
for token in sentence:
    print(token)
    print(token.embedding)

#Elmo Embedding加载训练
embedding = ELMoEmbeddings()
sentence = Sentence('The grass is green .')
embedding.embed(sentence)
for token in sentence:
    print(token)
    print(token.embedding)

#混合Embedding加载训练
stacked_embeddings = StackedEmbeddings([WordEmbeddings('model/glove.gensim'), FlairEmbeddings('model/news-forward-0.4.1.pt')])
sentence = Sentence('The grass is green .')
stacked_embeddings.embed(sentence)
for token in sentence:
    print(token)
    print(token.embedding)

#Character Embeddings和BytePairEmbeddings，无法翻墙则运行是下载会报错
embedding = CharacterEmbeddings()
sentence = Sentence('The grass is green .')
embedding.embed(sentence)
embedding = BytePairEmbeddings('en')
sentence = Sentence('The grass is green .')
embedding.embed(sentence)

