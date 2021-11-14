# -*- coding:utf-8 -*-
#神经网络的测试程序

import tensorflow as tf 
import hung_inference as hung_inference
import hung_train
import function_hung as hf 
import definition_constant as dc
import numpy as np


def evaluate(X, Y):
	with tf.Graph().as_default() as g:
		x = tf.placeholder(
			tf.float32, [None, hung_inference.INPUT_NODE], name="x-input")
		y_ = tf.placeholder(
			tf.float32, [None, hung_inference.OUTPUT_NODE], name="y-input")
		keep_prob = tf.placeholder("float")  

		y = hung_inference.inference(x, None, 1.0)

		correct_prediction = tf.equal(tf.argmax(y, 1), 
                                tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		variable_averages = tf.train.ExponentialMovingAverage(
			hung_train.MOVING_AVERAGAE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore() 
		saver = tf.train.Saver(variables_to_restore)


		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(
				hung_train.MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				global_step = ckpt.model_checkpoint_path\
					              .split('/')[-1].split('-')[-1]
				accuracy_score = sess.run(accuracy,
						                  feed_dict = {x: X, y_:Y})
				print("After %s training step(s), validation aaccuracy = %g" % (global_step, accuracy_score))
			else:
				print("No checkpoint file found")
				return
def main(argv=None):
	Xtest, Ytest_= hf.generate_data(dc.K, dc.num_test, seed=dc.testseed)
	Ytest=np.zeros((dc.num_test, dc.K))
	for i in range(dc.num_test):
		Ytest[i, int(Ytest_[i,0])]=1

	#print("X:",Xtest.shape,",Y:",Ytest.shape)
	evaluate(Xtest, Ytest)

if __name__ == "__main__":
	tf.app.run()