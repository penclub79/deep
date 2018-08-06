import  tensorflow as tf
import numpy as np

xy = np.loadtxt('zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

nb_classes = 7 #0 ~ 6까지 클래스

X = tf.placeholder(tf.float32, [None, 16]) #입력 16개
Y = tf.placeholder(tf.int32, [None, 1]) #출력 1개
#0 ~ 6

Y_one_hot = tf.one_hot(Y, nb_classes)  #one hot
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X, W) + b
#matmul 행렬 곱
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)

cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)    #0~6사이의 값(확률)
#tf.argmax함수는 텐서 내의 지정된 축에서 가장 높은 값의 인덱스를 반환합니다.
correct_prediction = tf.equal(prediction, tf.arg_max(Y_one_hot, 1))
#tf.equal에서는 예측 값과 정답이 같으면 True 아니면 False 값이 반환되는데,
# 이것을 float형으로 바꾸고 평균을 계산해 정확도를 구합니다.
# 정확도는 학습 데이터가 아닌 테스트 데이터를 사용해야합니다.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# tf.cast 로 입력 값을 부동소수점 값으로 변경합니다.
# 또 tf.cast 는 True, False 로 채워진 correct_prediction 텐서의 값을 숫자 값으로 변경하는 효과도 발휘
#학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={
                X: x_data, Y: y_data})
            print("Step : {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))
    #predict보기
    pred = sess.run(prediction, feed_dict={X: x_data})

    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y : {}".format(p == int(y), p, int(y)))
