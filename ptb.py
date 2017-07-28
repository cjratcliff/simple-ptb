from __future__ import division
from __future__ import print_function
import argparse
import time
import copy

import tensorflow as tf
import numpy as np

from ptb_reader import ptb_raw_data

# Adapted from https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py


class SmallConfig(object):
	max_grad_norm = 5
	num_layers = 2
	num_steps = 20
	hidden_size = 200
	max_epochs = 13
	keep_prob = 1.0
	batch_size = 20
	decay_lr_at = 4
	lr_decay = 0.5
	vocab_size = 10000


class MediumConfig(object):
	max_grad_norm = 5
	num_layers = 2
	num_steps = 35
	hidden_size = 650
	max_epochs = 39
	keep_prob = 0.5
	batch_size = 20
	decay_lr_at = 6
	lr_decay = 0.8	
	vocab_size = 10000


class LargeConfig(object):
	max_grad_norm = 10
	num_layers = 2
	num_steps = 35
	hidden_size = 1500
	max_epochs = 55
	keep_prob = 0.35
	batch_size = 20
	decay_lr_at = 14
	lr_decay = 1 / 1.15
	vocab_size = 10000
	

class PTBModel(object):
	def __init__(self,config):
		self.c = c = config
		
		self.x = tf.placeholder(tf.int32, [None, None], 'x')
		self.y = tf.placeholder(tf.int32, [None, None], 'y')
		self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
		self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

		with tf.device("/cpu:0"):
			embedding = tf.get_variable("embedding", [c.vocab_size, c.hidden_size], dtype=tf.float32)
			h = tf.nn.embedding_lookup(embedding, self.x)
			
		h = tf.nn.dropout(h, self.keep_prob)		
		h = tf.unstack(h, num=c.num_steps, axis=1)
		
		cells = [tf.contrib.rnn.LSTMCell(num_units = c.hidden_size) for i in range(c.num_layers)]				
		cells = [tf.contrib.rnn.DropoutWrapper(i, output_keep_prob=self.keep_prob) for i in cells]
		cells = tf.contrib.rnn.MultiRNNCell(cells)
		
		self.initial_state = cells.zero_state(self.batch_size, tf.float32)		
		
		outputs, self.final_state = tf.contrib.rnn.static_rnn(cells, h, initial_state=self.initial_state)
		h = tf.reshape(tf.stack(axis=1, values=outputs), [-1, c.hidden_size])
		
		logits = tf.contrib.layers.fully_connected(h, c.vocab_size, activation_fn=tf.identity)

		# Reshape logits to be a 3D tensor for sequence loss
		logits = tf.reshape(logits, [c.batch_size, c.num_steps, c.vocab_size])
		
		loss = tf.contrib.seq2seq.sequence_loss(
				logits,
				self.y,
				tf.ones([c.batch_size, c.num_steps], dtype=tf.float32),
				average_across_timesteps=False,
				average_across_batch=True
		)
		self.loss = tf.reduce_sum(loss)
		
		main_params = tf.trainable_variables()

		self.lr = tf.Variable(tf.constant(1.0),trainable=False)
		optimizer = tf.train.GradientDescentOptimizer(self.lr)

		grads = tf.gradients(self.loss, main_params)
			
		grads,_ = tf.clip_by_global_norm(grads, c.max_grad_norm)
		gv = zip(grads,main_params)
		self.train_step = optimizer.apply_gradients(gv)


	def fit(self, train_data, val_data, c, sess):
		c = self.c

		results = []
		for epoch in range(c.max_epochs):
			print("\nEpoch %d" % (epoch+1))
			start = time.time()
			
			# Decay the learning rate
			if epoch >= c.decay_lr_at:
				sess.run(tf.assign(self.lr,c.lr_decay*self.lr))
				print("Learning rate set to: %f" % sess.run(self.lr))
			
			train_perplexity = self.run_epoch(train_data, True, False, sess)
			print("Train perplexity: %.3f" % train_perplexity)
			
			val_perplexity = self.run_epoch(val_data, False, False, sess)
			print("Val perplexity: %.3f" % val_perplexity)
			
			print("Time taken: %.3f" % (time.time() - start))
			
			results.append([train_perplexity,val_perplexity])
			np.savetxt('results.csv', np.array(results), fmt='%5.5f', delimiter=',')


	def reshape_data(self,data):
		c = self.c
		num_batches = len(data) // c.batch_size
		data = data[0 : c.batch_size * num_batches]
		data = np.reshape(data,[c.batch_size, num_batches])
		return data
		
			
	def run_epoch(self, data, is_training, full_eval, sess):
		assert not(is_training and full_eval)
		c = copy.deepcopy(self.c)
		
		if full_eval: # Very slow so only used for the test set
			c.batch_size = 1
			c.num_steps = 1		

		data = self.reshape_data(data)
		
		total_loss = 0.0
		total_iters = 0.0
		
		state = sess.run(self.initial_state, feed_dict={self.batch_size: c.batch_size})
				
		for idx in range(0,data.shape[1],c.num_steps):
			batch_x = data[:, idx:idx+c.num_steps]
			batch_y = data[:, idx+1:idx+c.num_steps+1]
			
			if batch_x.shape != (c.batch_size,c.num_steps) or \
				batch_y.shape != (c.batch_size,c.num_steps):
				#print(batch_x.shape,batch_y.shape)
				continue
							
			feed_dict = {self.x: batch_x, 
						self.y: batch_y,
						self.batch_size: c.batch_size}
					
			for i, (c_state,h_state) in enumerate(self.initial_state):
				feed_dict[c_state] = state[i].c
				feed_dict[h_state] = state[i].h	

			if is_training:
				feed_dict[self.keep_prob] = c.keep_prob
			else:
				feed_dict[self.keep_prob] = 1.0

			if is_training:
				_,loss,state = sess.run([self.train_step, self.loss, self.final_state], feed_dict)			
			else:
				loss,state = sess.run([self.loss,self.final_state], feed_dict)
			
			total_loss += loss
			total_iters += c.num_steps

		return np.exp(total_loss/total_iters)
		

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--size', type=str, help='small, medium or large', required=True)
	args = parser.parse_args()

	X_train, X_valid, X_test, vocab = ptb_raw_data()
	
	print("\nData loaded")
	print("Training set: %d words" % len(X_train))
	print("Validation set: %d words" % len(X_valid))
	print("Test set: %d words" % len(X_test))
	print("Vocab size: %d words\n" % len(vocab))
		
	if args.size == 'small':
		c = SmallConfig()
	elif args.size == 'medium':
		c = MediumConfig()	
	elif args.size == 'large':
		c = LargeConfig()	
	else:
		raise ValueError("Invalid value for size argument")

	m = PTBModel(c)
	
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	
	m.fit(X_train,X_valid,c,sess)	
	m.run_epoch(X_test, False, True, sess)
		
if __name__ == "__main__":
	main()
