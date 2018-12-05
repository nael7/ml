#############################################################################
# multi-variable linear regression example using TensorFlow library.
#############################################################################
# Author: Geol Choi, phD.(cinema4dr12@gmail.com)
# Date: July 5, 2017
#############################################################################
 
## import libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import pandas as pd
 
## define parameters
learning_rate = 0.001
training_epochs = 4000
display_step = 200

def preprocess_features(data):
  """Prepares input features from California housing data set.

  Args:
    grade: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """
  selected_features = data[
    ["Value"
     ,"CPIAUCSL"
     ,"LNS14000024"
     ,"value2"
     ]]
  
  #selected_features = grade[
  #  [feature_str]]
  
  processed_features = selected_features.copy()
  # Create a synthetic feature.
  #processed_features["rooms_per_person"] = (
  #  grade["total_rooms"] /
  #  grade["population"])
  return processed_features

def preprocess_targets(data):
  """Prepares target features (i.e., labels) from California housing data set.

  Args:
    grade: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets = pd.DataFrame()
  # Scale the target to be in units of thousands of dollars.
  output_targets["Adj Close"] = (
    data["Adj Close"])
  return output_targets
 
## loading from data file
#xy = np.loadtxt('./iqsize.tsv', delimiter='\t', dtype=np.float32)

#index를 날짜로 생성후에 일단위 데이터를 월단위로 변환
data1 = pd.read_csv("./^DJI.csv", sep=",", index_col="Date", parse_dates=True)
data1 = data1.resample(rule='M').mean()
print(data1)

#index를 날짜로 생성후에 일단위 데이터를 월단위로 변환
data2 = pd.read_csv("./DGS10.csv", sep=",", index_col="Date", parse_dates=True)

#결측값 제거 및 형변환 (데이터에 .가 있어 문자형으로 들어옴)
data2 = data2[data2.Value != '.']
data2[['Value']] = data2[['Value']].astype(float)

data2 = data2.resample(rule='M').mean()
print(data2)

data3 = pd.read_csv("./CPIAUCSL.csv", sep=",", index_col="Date", parse_dates=True)
#날짜가 해당월의 첫날이라 하루전으로 시프트
data3 = data3.shift(-1, freq="D")
print(data3)

data4 = pd.read_csv("./LNS14000024.csv", sep=",", index_col="Date", parse_dates=True)
#날짜가 해당월의 첫날이라 하루전으로 시프트
data4 = data4.shift(-1, freq="D")
print(data4)

data5 = pd.read_csv("./fed-funds-rate-historical-chart.csv", sep=",", index_col="Date", parse_dates=True)
#날짜가 해당월의 첫날이라 하루전으로 시프트
data5 = data5.resample(rule='M').mean()
print(data5)

#Date값을 키로 데이터프레임 머지
data = pd.merge(data1, data2, on="Date")
data = pd.merge(data, data3, on="Date")
data = pd.merge(data, data4, on="Date")
data = pd.merge(data, data5, on="Date")

print(data)

data = data.reindex(
     np.random.permutation(data.index))

#수치를 나추어서 계산이 용이하게
data["Adj Close"] /= 1000.0
data["CPIAUCSL"] /= 10.0

training_examples = preprocess_features(data.head(200))
print(training_examples.describe())

training_targets = preprocess_targets(data.head(200))
print(training_targets.describe())
 
#y_data = xy[:, 0]
#x1_data = xy[:,1]
#x2_data = xy[:,2]
#x3_data = xy[:,3]
#x4_data = xy[:,4]

y_data = training_targets["Adj Close"]
 
x1_data = training_examples["Value"]
x2_data = training_examples["CPIAUCSL"]
x3_data = training_examples["LNS14000024"]
x4_data = training_examples["value2"]
 
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
 
hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + x4 * w4 + b
print(hypothesis)
 
## cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
 
## minimize. Need a very small learning rate for this data set
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)
 
## initializes global variables in the graph.
tf.set_random_seed(0)
init = tf.global_variables_initializer()
 
## initialize variable for plotting
cost_res = np.zeros(shape=(training_epochs,2))

save_file = 'C:/Temp/model.ckpt'
saver = tf.train.Saver()

## launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    # fit all training data
    for epoch in range(training_epochs):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, x4: x4_data, Y: y_data})
        cost_res[epoch, 0] = epoch
        cost_res[epoch, 1] = np.log(cost_val)
        
        if epoch % display_step == 0:
            print("Epoch: ", (epoch+1), " , Cost: ", cost_val)
            
    # graphic display
    plt.plot(cost_res[:,0], cost_res[:,1], 'ro', label='Cost Function Value as Epoch Proceeds')
    plt.show()
    
    #plt.plot(x0_data, x1_data, 'ro')
    #plt.plot(x2_data, y_data, 'bo')
    #plt.plot(x3_data, y_data, 'go')
    #plt.plot(x4_data, y_data, 'yo')
    #plt.plot(y_data, sess.run(w1) * x1_data + sess.run(w2) * x2_data + sess.run(w3) * x3_data + sess.run(w4) * x4_data + sess.run(b))
    #plt.legend()
    #plt.show()
    
    # save
    saver.save(sess, save_file)
    
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
    
    ## compare results
    for i in range(35):
        print(y_data[i], " , ", hy_val[i])
 
