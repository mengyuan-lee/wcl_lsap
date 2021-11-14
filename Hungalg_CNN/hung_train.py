# -*- coding:utf-8 -*-
#神经网络的训练程序

import os
import tensorflow as tf 
import hung_inference as hung_inference
import function_hung as hf 
import definition_constant as dc
import numpy as np
import time


BATCH_SIZE = 2048
LEARNING_RATE_BASE = 0.0001
LEARNING_RATE_DECAY = 0.99
REGULARAZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGAE_DECAY = 0.99 
TRAINTESTSPLIT = 0.1
MODEL_SAVE_PATH = "./hung_model/"
MODEL_NAME = "model.ckpt"

def train(X, Y):
	with tf.Graph().as_default() as g:
		num_total = X.shape[0]                        # number of total samples
		print(num_total)
		num_val = int(num_total * TRAINTESTSPLIT)     # number of validation samples
		num_train = num_total - num_val               # number of training samples
		X_train = X[0:num_train, :]    # training data
		Y_train = Y[0:num_train, :]   # training label
		X_val = X[num_train:num_total, :] # validation data
		Y_val = Y[num_train:num_total, :] # validation label
		
		x = tf.placeholder(
			tf.float32, [None, hung_inference.INPUT_NODE], name="x-input") 
		y_ = tf.placeholder(
			tf.float32, [None, hung_inference.OUTPUT_NODE], name="y-input")
		keep_prob = tf.placeholder("float")  


		regularizer = tf.contrib.layers.l2_regularizer(REGULARAZATION_RATE)
		y = hung_inference.inference(x, regularizer, 1.0)
		global_steps = tf.Variable(0, trainable=False, name='global_steps')

		correct_prediction = tf.equal(tf.argmax(y, 1), 
                                tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		variable_averages = tf.train.ExponentialMovingAverage(
			MOVING_AVERAGAE_DECAY, global_steps)  
		variable_averages_op = variable_averages.apply(
			tf.trainable_variables())

		print(y_.shape)


		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits = y, 
			labels = tf.argmax(y_, 1))
		cross_entropy_mean = tf.reduce_mean(cross_entropy) 
		loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		learning_rate = tf.train.exponential_decay(
			LEARNING_RATE_BASE,
			global_steps, 
			num_total / BATCH_SIZE, 
			LEARNING_RATE_DECAY)
		
		total_batch = int(num_train / BATCH_SIZE)

		#train_step = tf.train.GradientDescentOptimizer(learning_rate)\
	    #       	  .minimize(loss, global_step=global_steps)
		train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, global_step=global_steps)
		with tf.control_dependencies([train_step, variable_averages_op]):
			train_op = tf.no_op(name="train")

		if not os.path.exists(MODEL_SAVE_PATH):
			os.makedirs(MODEL_SAVE_PATH) 
		saver = tf.train.Saver() 
		with tf.Session() as sess:
			
			ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				print('Checkpoint file found')
			else:
				print('No checkpoint file found')
				tf.global_variables_initializer().run()


			for epoch in range(dc.train_epochs):
				for i in range(total_batch):
					idx = np.random.randint(num_train,size=BATCH_SIZE)
					_, loss_train, step= sess.run([train_op, loss, global_steps], feed_dict = {x: X_train[idx, :], y_: Y_train[idx, :]})

					if epoch%(int(dc.train_epochs/10))==0:
						loss_val = sess.run(loss, feed_dict={x: X_val, y_: Y_val})
						print("epoch: %d, Training step: %d, Loss on trainning: %g, Loss on validation: %g" % (epoch, step, loss_train, loss_val))
						saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = step)
						train_accuracy = sess.run(accuracy, feed_dict = {x: X_train[idx, :], y_: Y_train[idx, :]})
						valid_accuracy = sess.run(accuracy, feed_dict = {x: X_val, y_: Y_val})
						print("train accuracy = %g, validation aaccuracy = %g" % (train_accuracy, valid_accuracy))

		

def main(argv=None):
	X, Y_= hf.generate_data(dc.K, dc.num_train, seed=dc.trainseed)
	Y=np.zeros((dc.num_train, dc.K))
	for i in range(dc.num_train):
		Y[i, int(Y_[i,0])]=1.0

	print(X.shape, Y.shape)
	train(X, Y)

if __name__ == "__main__":
	tf.app.run()


