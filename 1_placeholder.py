import tensorflow as tf

a = tf.placeholder(tf.float32) #딕셔너리형태로 들어간다. #변수이다. 값입력을 유연하게 할수 있다.
b = tf.placeholder(tf.float32)

adder_node = a + b
sess = tf.Session()
print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4.3]}))

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, feed_dict={a: 3, b: 4.5}))

