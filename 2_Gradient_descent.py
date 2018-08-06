import tensorflow as tf

X = [1, 2, 3]
Y = [1, 2, 3]
# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)
W = tf.Variable(-4.0) # 5.일경우
# W = tf.Variable(tf.random_normal([1]), name = 'weight')
# b = tf.Variable(tf.random_normal([1]), name = 'bias')
hypothesis = X * W
# hypothesis = X * W + b

# Our hypothesis XW + b
cost = tf.reduce_mean(tf.square(hypothesis-Y))
# cost = tf.reduce_mean(tf.square(hypothesis-y_train))

#Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) #learning_rate(매우중요)
#learning_rate :
train = optimizer.minimize(cost)

#Launch the grath in a session
sess = tf.Session()

#initializes gglobal variables in the graph
sess.run(tf.global_variables_initializer())

#Fit the line
for step in range(100):
    # sess.run(train)
    # if step % 20 == 0:
    print(step, sess.run(W))
    sess.run(train)
# for step in range(2001):
#     cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X:[1,2,3,4,5], Y:[2.1,3.1,4.1,5.1,6.1]})
#     if step % 20 == 0:
#         print(step, cost_val, W_val, b_val)
