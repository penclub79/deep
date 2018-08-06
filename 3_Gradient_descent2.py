import tensorflow as tf
import matplotlib.pyplot as plt

#tf Graph Input
X = [1,2,3]
Y = [1,2,3]
W = tf.placeholder(tf.float32)

#Our hypothesis for linear model X * W
hypothesis = X * W
# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#Launch the gragh in a session.
sess = tf.Session()

# Variables for plotting cost function
W_history = []
cost_history = []
for i in range(-30, 50): #0.1을 곱함으로써 X축은  -3 ~ 5이다.
    curr_W = i * 0.1
    curr_cost = sess.run(cost, feed_dict={W:curr_W})
    W_history.append(curr_W)
    cost_history.append(curr_cost)

#Show the cost function
plt.plot(W_history, cost_history)
plt.show()
