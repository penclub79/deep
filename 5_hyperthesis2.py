import tensorflow as tf

#5x3행렬과 5x1행렬을 곱한 결과 분석 H(X) = XW
x_data = [[73., 80., 75.],
        [93., 88., 93.],
        [89., 91., 90.],
        [96., 98., 100.],
        [73., 66., 70.]]

y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]
# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)
# W = tf.Variable(-4.0) # 5.일경우
X = tf.placeholder(tf.float32, shape = [None, 3])
Y = tf.placeholder(tf.float32, shape = [None, 1])
W = tf.Variable(tf.random_normal([3, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')
hypothesis = tf.matmul(X, W) + b
# hypothesis = X * W + b

# Our hypothesis XW + b
cost = tf.reduce_mean(tf.square(hypothesis-Y))
# cost = tf.reduce_mean(tf.square(hypothesis-y_train))

#Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5) #learning_rate(매우중요)
#learning_rate :
train = optimizer.minimize(cost)

#Launch the grath in a session
sess = tf.Session()

#initializes gglobal variables in the graph
sess.run(tf.global_variables_initializer())

for step in range(2001):
#sess.run을 통해 hy 와 cost 그래프를 계산합니다.
#이때 가설 수식에 넣어야 할 실제값을 feed_dict을 통해 전달합니다.
    cost_val, hy_val = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost ", cost_val, "\nPrediction:\n", hy_val)
