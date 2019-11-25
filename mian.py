# coding: utf-8
import tensorflow as tf
# from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import numpy as np
# 获取数据
# boston = load_boston()
# X = scale(boston.data)
# y = scale(boston.target.reshape((-1,1)))
from itertools import islice
data_X = []
data_Y = []
with open('data/boston_house_prices.csv') as f:
    for line in islice(f, 0, None):
        line = line.split(',')
        data_X.append(line[:-1])
        data_Y.append(line[-1:])
# 转换为np array
data_X = np.array(data_X, dtype='float32')
data_Y = np.array(data_Y, dtype='float32')
print('data shape', data_X.shape, data_Y.shape)
print('data_x shape[1]', data_X.shape[1])
X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.5, random_state=1)
mean = X_train.mean(axis = 0)
std = X_train.std(axis = 0)
X_train -= mean
X_train /= std
X_test -= mean
X_test /= std

# 使网络更灵活
def add_layer(inputs,input_size,output_size,activation_function=None):
    with tf.variable_scope("Weights"):
        Weights = tf.Variable(tf.random_normal(shape=[input_size, output_size]), name="weights")
    with tf.variable_scope("biases"):
        biases = tf.Variable(tf.zeros(shape=[1,output_size]) ,name="biases")
    with tf.name_scope("Wx_plus_b"):
        Wx_plus_b = tf.matmul(inputs,Weights) + biases
    with tf.name_scope("dropout"):
        Wx_plus_b = tf.nn.dropout(Wx_plus_b,keep_prob=keep_prob_s)
    if activation_function is None:
        return Wx_plus_b
    else:
        with tf.name_scope("activation_function"):
            return activation_function(Wx_plus_b)


# 参数设置
xs = tf.placeholder(shape=[None,X_train.shape[1]],dtype=tf.float32,name="inputs")
ys = tf.placeholder(shape=[None,1],dtype=tf.float32,name="y_true")
keep_prob_s = tf.placeholder(dtype=tf.float32)

with tf.name_scope("layer_1"):
    l1 = add_layer(xs,13,200,activation_function=tf.nn.relu)
with tf.name_scope("layer_2"):
    l2 = add_layer(l1,200,300,activation_function=tf.nn.relu)
# with tf.name_scope("layer_3"):
#     l3 = add_layer(l2,500,64,activation_function=tf.nn.relu)
with tf.name_scope("y_pred"):
    pred = add_layer(l2,300,1)

# 这里多于的操作，是为了保存pred的操作，做恢复用。我只知道这个笨方法。
pred = tf.add(pred,0,name='pred')

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - pred),reduction_indices=[1]))  # mse
    tf.summary.scalar("loss",tensor=loss)
with tf.name_scope("train"):
    #train_op =tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
    train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

# 可视化
# draw pics
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(range(50),y_train[0:50],'b')  #展示前50个数据
ax.set_ylim([-2,5])
plt.ion()
plt.show()

# parameters
keep_prob = 0.9  # 防止过拟合，取值一般在0.5到0.8。
ITER = 5000  # 训练次数


# 训练定义
def fit(X, y, ax, n, keep_prob):
    print(y)
    init = tf.global_variables_initializer()
    feed_dict_train = {ys: y, xs: X, keep_prob_s: keep_prob}
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logdir="nn_boston_log", graph=sess.graph)  #写tensorbord
        sess.run(init)
        for i in range(n):
            _loss, _ = sess.run([loss, train_op], feed_dict=feed_dict_train)
            if i % 100 == 0:
                print("epoch:%d\tloss:%.5f" % (i, _loss))
                y_pred = sess.run(pred, feed_dict=feed_dict_train)
                rs = sess.run(merged, feed_dict=feed_dict_train)
                writer.add_summary(summary=rs, global_step=i)  # 写tensorbord
                saver.save(sess=sess, save_path="nn_boston_model/nn_boston.model", global_step=i) # 保存模型
                try:
                    ax.lines.remove(lines[0])
                except:
                    pass
                lines = ax.plot(range(50), y_pred[0:50], 'r--')
                plt.pause(1)

        saver.save(sess=sess, save_path="nn_boston_model/nn_boston.model", global_step=n)  # 保存模型


def test(X, y,  keep_prob):
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
        batch_size = 1
        saver.restore(sess, tf.train.latest_checkpoint("nn_boston_model/"))  # 加载变量值
        print('finish loading model!')
        # test
        test_total_batch = int(len(y_test) / batch_size)
        test_accuracy_list = []
        test_loss_list = []
        y_truly = []
        y_pre = []
        x_test_batch_list = []
        for j in range(test_total_batch):
            x_test_batch, y_test_batch = get_batch(X, y, batch_size, j, test_total_batch)
            test_accuracy, test_loss = sess.run([pred, loss],
                                                feed_dict={xs: x_test_batch, ys: y_test_batch, keep_prob_s: keep_prob})
            test_loss_list.append(test_loss)
            y_truly.append(y_test_batch[0][0])
            y_pre.append(test_accuracy[0][0])
            x_test_batch_list.append(x_test_batch)
        print('test_loss:' + str(np.mean(test_loss_list)))
        print(y_pre[0:50])
        print(y_truly[0:50])
        plt.ylim((-5,50))
        plt.plot(range(50),y_truly[:50], 'r')
        plt.plot(range(50),y_pre[:50], 'b')
        plt.show()


def get_batch(image, label, batch_size, now_batch, total_batch):
    if now_batch < total_batch:
        image_batch = image[now_batch*batch_size:(now_batch+1)*batch_size]
        label_batch = label[now_batch*batch_size:(now_batch+1)*batch_size]
    else:
        image_batch = image[now_batch*batch_size:]
        label_batch = label[now_batch*batch_size:]
    return image_batch, label_batch


fit(X=X_train, y=y_train, n=ITER, keep_prob=keep_prob, ax=ax)
test(X=X_test, y=y_test, keep_prob=1)