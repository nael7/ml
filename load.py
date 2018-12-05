import tensorflow as tf
import numpy as np

# The file path to save the data
save_file = 'C:/Temp/model.ckpt'
 
# Remove the previous weights and bias
tf.reset_default_graph()

## placeholders for a tensor that will be always fed.
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
x4 = tf.placeholder(tf.float32)
 
Y = tf.placeholder(tf.float32)
 
## define varaibles & hypothesis
w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
w4 = tf.Variable(tf.random_normal([1]), name='weight4')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 가설을 설정합니다. 
hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + x4 * w4 + b
 
# Class used to save and/or restore Tensor Variables
saver = tf.train.Saver()
 
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    # Load the weights and bias
    saver.restore(sess, save_file)
 
    # Show the values of weights and bias
    print('weight1:')
    print(sess.run(w1))
    print('weight2:')
    print(sess.run(w2))
    print('weight3:')
    print(sess.run(w3))
    print('weight4:')
    print(sess.run(w4))
    print('bias:')
    print(sess.run(b))   
    
    x1_data = float('3.10') # 채권
    x2_data = float('22') # 물가지수 1/10
    x3_data = float('3.5') # 실업율
    x4_data = float('2.25') # 금리

    # 배추 가격 변수를 선언합니다.
        #["Value"
#     ,"CPIAUCSL"
 #    ,"LNS14000024"
  #   ,"value2"
    price = 0

    # 입력된 파라미터를 배열 형태로 준비합니다.
    #data = ((avg_temp, min_temp, max_temp, rain_fall), (0, 0, 0, 0))
    #arr = np.array(data, dtype=np.float32)

    # 입력 값을 토대로 예측 값을 찾아냅니다.
    #x_data = arr[0:4]
    dict = sess.run(hypothesis, feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, x4: x4_data})

    # 결과 배추 가격을 저장합니다.
    price = dict[0]
        
    print('price:')
    print(price)
    
    sess.close()
