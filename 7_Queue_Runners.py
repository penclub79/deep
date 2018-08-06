import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

filename_queue = tf.train.string_input_producer(#데이터를 저장하는 컴포넌트
    ['test-score.csv'], shuffle=False, name='filename_queue')
print('------',filename_queue) #filename Queue에서 데이터를 섞지 않고 쌓는다.
reader = tf.TextLineReader() #텍스트파일에서, 한줄씩 읽어서 문자열을 리턴(데이터를 읽어오는 컴포넌트)
print('reader=======', reader)
key, value = reader.read(filename_queue)
#filename_queue로 부터 파일을 읽게하면 value에 파일에서 읽은 값이 리턴이 된다.


# Default values, in case of empty columns. Also specifies the type of the
#읽은 데이터를 디코딩 하기(Decoder)
#reader에서 읽은 값은 파일의 원시 데이터(raw)이다. 아직 해석(파싱)이 된데이터가 아님.
# decoded result.
record_defaults = [[0.], [0.], [0.], [0.]]
print(record_defaults)
#Reader와 마찬가지로, Decoder 역시 미리 정해진 Decoder 타입이 있는데,
# JSON,CSV 등 여러가지 데이타 포맷에 대한 Decoder를 지원한다.

xy = tf.decode_csv(value, record_defaults=record_defaults)
print('xy=',xy)
#위의 CSV 문자열을 csv 디코더를 이용하여 파싱해보자
#csv decoder를 사용하기 위해서는 각 필드의 디폴트 값을 지정해줘야한다.
#record_default는 각 필드의 디폴트 값을 지정해주는건 물론이고, 각 필드의 데이터 타입(string, int, float etc)를 정의
#예를 들어(167c9599-c97d-4d42-bdb1-027ddaed07c0,1,2016,REG,3:54)의 데이터를 csv파일을 읽어왔다면
#디폴트값은 csv 데이터에서 해당 필드가 비워져 있을때 채워 진다. 첫번째 필드는 string형이다. 167c9599-c97d-4d42-bdb1-027ddaed07c0
#두번째는 int형 1 세번째도 int형 2016.

#이 디폴트값 세팅을 가지고 tf.decode_csv를 이용하여 파싱한다.



# collect batches of csv in
#여러개의 파일을 병렬로 처리하고자 한다면, [file_name]부분에 리스트 형으로 여러개의 파일 목록을 지정해주면 된다
train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

#위의 예제에서는 batch_size를 10으로 해줬기 때문에, batch_year = [ 1900,1901….,1909]  와 같은 형태로 10개의 년도를 하나의 텐서에 묶어서 리턴해준다.
#즉 입력 텐서의 shape이  [x,y,z] 일 경우 tf.train.batch를 통한 출력은 [batch_size,x,y,z] 가 된다.(이 부분이 핵심)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
print(X)
Y = tf.placeholder(tf.float32, shape=[None, 1])
print(Y)
W = tf.Variable(tf.random_normal([3, 1]), name='weight')
print(W)
b = tf.Variable(tf.random_normal([1]), name='bias')
print(b)

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
# x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 10 == 0:
        print('step=',step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

#모든 작업이 끝났으면 아래와 같이 Queue runner를 정지 시킨다
coord.request_stop()
coord.join(threads)

# Ask my score
print("Your score will be ",
      sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))

print("Other scores will be ",
      sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))