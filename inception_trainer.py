import tensorflow as tf
import prettytensor as pt
import numpy as np
from inception_tensors import *

def vectorize_id(idnum, labels_len=256):
	vec = np.zeros(labels_len)
	vec[int(idnum) - 1] = 1
	return vec

def random_batch(train_set, num_labels=256, batch_size = 100):
	idx = list(np.random.choice(len(train_set), batch_size, replace=False))
	batch = []
	for ii, item in enumerate(train_set):
		if ii in idx:
			batch.append(item)
	paths, names, ids = zip(*batch)

	x_batch = [classify(one) for one in paths]
	y_batch = [vectorize_id(one, num_labels) for one in ids]

	return x_batch, y_batch

def train(train_set, labels, transfer_len=2048):
	# In/out
	with tf.name_scope('InputArgs'):
		x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='TransferExample')
		y_true = tf.placeholder(tf.float32, shape=[None, len(labels)], name='LabelVector')
		x_pretty = pt.wrap(x)

	# Connections
	# Does softmax classifier use cross entropy???
	with tf.name_scope('TransferLayer'):
		with pt.defaults_scope(activation_fn=tf.nn.relu):
			y_guess, loss = x_pretty                                        \
				.fully_connected(size=transfer_len, name='FullLayer1')    \
				.softmax_classifier(num_classes=len(labels), labels=y_true, name='SoftMaxLayer1')

	with tf.name_scope('Gradient'):
		train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

	with tf.name_scope('BatchAccuracy'):
		y_guess_index = tf.argmax(y_guess, axis=1)
		y_guess_is_correct = tf.equal(y_guess_index, tf.argmax(y_true, axis=1))
		accuracy = tf.reduce_mean(tf.cast(y_guess_is_correct, tf.float32))

	print '3. Defined learning graph'

	tr_session = tf.Session()
	train_writer = tf.summary.FileWriter('logs/train', tr_session.graph)
	tr_session.run(tf.global_variables_initializer())

	num_iterations = 100
	for ii in range(num_iterations):
		x_batch, y_true_batch = random_batch(train_set, len(labels), batch_size=4)

		train_input = {x: x_batch, y_true: y_true_batch}
		_ = tr_session.run([train_step], feed_dict=train_input)

		if (ii % 4 == 0) or (ii == num_iterations - 1):
			batch_acc, batch_results = tr_session.run([accuracy, y_guess_index], feed_dict=train_input)
			print len(x_batch), len(batch_results)
			msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
			print(msg.format(ii, batch_acc))
