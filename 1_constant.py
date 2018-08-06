import tensorflow as tf

node1 = tf.constant(3.4, tf.float32) #배열 형태 값으로 됩니당 # 상수값: 값이 고정
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print("node1 : ", node1, "node2 : ", node2)
print("node3 : ", node3)

sess = tf.Session()

print("sess.run(node1, node2) : ", sess.run([node1, node2]))
print("sess.run(node3) : ", sess.run([node3]))
