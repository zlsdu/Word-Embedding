import tensorflow as tf
import tensorflow_hub as hub


if __name__ == '__main__':
    # 使用tensorflow_hub加载elmo模型进行词向量预训练
    elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=True)
    # 三条英文训练数据
    sentence_lists = ["My name is Lei", "I am studying machine learning", "I am fine thanks"]
    output = elmo(sentence_lists, as_dict=True)
    print(output)
    # 启动图运行elmo模型得出词向量
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    elmo_embedding = sess.run(output['elmo'])
    print(elmo_embedding)
