import tensorflow as tf
import matplotlib.pyplot as plt
#일렬로 행렬했을시
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]

X1 = tf.placeholder(tf.float32)
X2 = tf.placeholder(tf.float32)
X3 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([1]), name='weight1')
W2 = tf.Variable(tf.random_normal([1]), name='weight2')
W3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name= 'bias')
hypothesis = X1 * W1 + X2 * W2 + X3 * W3 + b

#Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 1e-5)
train = optimizer.minimize(cost)

#Launch the graph in a session.
sess = tf.Session()

#Initialzes global variables in the graph(# 세션을 생성하고 초기화 합니다.)
sess.run(tf.global_variables_initializer())


for step in range(2001):
#sess.run을 통해 hy 와 cost 그래프를 계산합니다.
#이때 가설 수식에 넣어야 할 실제값을 feed_dict을 통해 전달합니다.
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X1:x1_data, X2:x2_data, X3:x3_data, Y:y_data})
    if step % 10 == 0:
        print(step, "Cost ", cost_val, "\nPrediction:\n", hy_val)

plt.plot()
