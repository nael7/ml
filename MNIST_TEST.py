import os
import cv2
from numpy import array
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np

##이미지 검증
INPUT_DIR = './MNIST_CNN/'
input_folder_list = array(os.listdir(INPUT_DIR))
    
print(input_folder_list)
     
test_input = []
     
for index in range(len(input_folder_list)):
    path = os.path.join(INPUT_DIR, input_folder_list[index])
    path = path + '/'
    img_list = os.listdir(path)
    for img in img_list:
        img_path = os.path.join(path, img)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        test_input.append([np.array(img)])
        #train_label.append([np.array(onehot_encoded[index])])

print(test_input)
     
test_input = np.reshape(test_input, (-1, 784))
print(test_input)

test_input = np.array(test_input).astype(np.float)
#np.save("test_data.npy", test_input)
print(test_input)
   
# The file path to save the data
save_file = 'C:/Temp/model1.ckpt'
 
# tensorflow graph input
X = tf.placeholder('float', [1, 784]) # mnist data image of shape 28 * 28 = 784
Y = tf.placeholder('float', [1, 10]) # 0-9 digits recognition = > 10 classes

# set dropout rate
dropout_rate = tf.placeholder("float")

# set model weights
W1 = tf.get_variable("W1", shape=[784, 256])
W2 = tf.get_variable("W2", shape=[256, 256])
W3 = tf.get_variable("W3", shape=[256, 256])
W4 = tf.get_variable("W4", shape=[256, 256])
W5 = tf.get_variable("W5", shape=[256, 10])

B1 = tf.Variable(tf.random_normal([256]))
B2 = tf.Variable(tf.random_normal([256]))
B3 = tf.Variable(tf.random_normal([256]))
B4 = tf.Variable(tf.random_normal([256]))
B5 = tf.Variable(tf.random_normal([10]))

# Construct model
_L1 = tf.nn.relu(tf.add(tf.matmul(X,W1),B1))
L1 = tf.nn.dropout(_L1, dropout_rate)
_L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2),B2)) # Hidden layer with ReLU activation
L2 = tf.nn.dropout(_L2, dropout_rate)
_L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3),B3)) # Hidden layer with ReLU activation
L3 = tf.nn.dropout(_L3, dropout_rate)
_L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4),B4)) # Hidden layer with ReLU activation
L4 = tf.nn.dropout(_L4, dropout_rate)

hypothesis = tf.add(tf.matmul(L4, W5), B5) # No need to use softmax here

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits=hypothesis)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost) # Adam Optimizer
 
# Class used to save and/or restore Tensor Variables
saver = tf.train.Saver()
 
with tf.Session() as sess:
    
    sess.run(tf.initialize_all_variables())
    
    # Load the weights and bias
    saver.restore(sess, save_file)
 
    # Show the values of weights and bias
    print('weight1:')
    print(sess.run(W1))
    print('weight2:')
    print(sess.run(W2))
    print('weight3:')
    print(sess.run(W3))
    print('weight4:')
    print(sess.run(W4))
    print('weight5:')
    print(sess.run(W5))   

    # 배추 가격 변수를 선언합니다.
        #["Value"
#     ,"CPIAUCSL"
 #    ,"LNS14000024"
  #   ,"value2"
    #price = 0

    # 입력된 파라미터를 배열 형태로 준비합니다.
    #data = ((avg_temp, min_temp, max_temp, rain_fall), (0, 0, 0, 0))
    #arr = np.array(data, dtype=np.float32)

    # 입력 값을 토대로 예측 값을 찾아냅니다.
    #x_data = arr[0:4]
    #dict = sess.run(hypothesis, feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, x4: x4_data})
    
    #dict = sess.run(hypothesis, feed_dict={X: test_input[0]})
    #Y = sess.run(optimizer, feed_dict={X: test_input, dropout_rate: 0.7})
    Y = sess.run(hypothesis, feed_dict={X: test_input, dropout_rate: 0.7})

    max = 0
    dist = 0
    index = 0
    for x in Y[0]:
        if x > max: 
            max = x
            dist = index
        index = index + 1 
    
    print('label:')
    print(dist)

    # 결과 배추 가격을 저장합니다.
    #price = Y[0]
        
    #print('price:')
    #print(price)
    
    sess.close()
    
# 이미지 표시
f, a = plt.subplots(1, 10, figsize=(10, 2))
a[0].imshow(np.reshape(test_input[0], (28, 28)))
f.show()
plt.draw()
plt.waitforbuttonpress() # 버튼을 누를때까지 작업 정지

print('end')

