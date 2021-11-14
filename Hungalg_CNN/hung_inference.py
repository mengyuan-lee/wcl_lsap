# -*- coding:utf-8 -*-
#定义了前向传播的过程以及神经网络的参数

import tensorflow as tf 

#定义神经网络结果相关的参数
INPUT_NODE = 16
OUTPUT_NODE = 4

def weight_variable(name, shape, regularizer):
	weights = tf.get_variable(
		name, shape,
		initializer=tf.truncated_normal_initializer(stddev=0.1))

	if regularizer != None:
		tf.add_to_collection("losses",regularizer(weights))
	return weights 

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	
def inference(input_tensor, regularizer, keep_prob): 
	x = tf.reshape(input_tensor, [-1, 4, 4, 1])

	W_conv1 = weight_variable("W_conv1",[1, 1, 1, 2], regularizer)  # 卷积是在每个5*5的patch中算出32个特征，分别是patch大小，输入通道数目，输出通道数目
	b_conv1 = bias_variable([2])  
  
	h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)  #第一个卷积层
	#h_pool1 = max_pool_2x2(h_conv1)                     #第一个池化层

	W_conv2 = weight_variable("W_conv2",[1, 1, 2, 4], regularizer)  
	b_conv2 = bias_variable([4])  
  
	h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)  #第二个卷积层
	#h_pool2 = max_pool_2x2(h_conv2)                           #第二个池化层

	W_fc1 = weight_variable("W_fc1",[4 * 4 * 4, 256], regularizer)  
	b_fc1 = bias_variable([256])  
  
	h_pool2_flat = tf.reshape(h_conv2, [-1, 4*4*4])           #reshape成向量
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) #第一个全连接层

	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  #dropout层 
  
	W_fc2 = weight_variable("W_fc2",[256, 4],regularizer)  
	b_fc2 = bias_variable([4])  
	y_conv=tf.matmul(h_fc1_drop, W_fc2) + b_fc2 
	
	return y_conv

