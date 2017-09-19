import tensorflow as tf
import prettytensor as pt
import numpy as np
from inception_tensors import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import datetime as dt

def vectorize_id(idnum, labels_len=256):
	vec = np.zeros(labels_len)
	vec[int(idnum) - 1] = 1
	return vec

def random_batch(train_set, num_labels=256, batch_size = 100, shuffle=True):
	bucket_size = int(np.floor(batch_size / num_labels))
	# print 'BUCK', bucket_size, batch_size, num_labels
	batch = []
	for category in train_set:
		if shuffle:
			idx = list(np.random.choice(len(category), bucket_size, replace=False))
		else: idx = [index for index, _ in enumerate(category[:bucket_size])]

		# print 'INDEX', idx
		for ii, item in enumerate(category):
			if ii in idx:
				batch.append(item)

	# print batch[0]
	paths, names, ids = zip(*batch)

	x_batch = [classify(one) for one in paths]
	y_batch = [vectorize_id(one, num_labels) for one in ids]

	return x_batch, y_batch, (paths,)

def split_data(examples, reserved=2):
	train_set = []
	test_set = []
	for category in examples:
		train_set.append(category[:-reserved])
		test_set.append(category[-reserved:])
	return train_set, test_set

def train(examples, labels, transfer_len=2048):
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

	plt.ion()
	fig = None
	train_batch_size = 9
	plotdim = int(np.sqrt(train_batch_size))

	train_set, test_set = split_data(examples, reserved=plotdim)

	t1 = dt.datetime.now()

	for ii in range(num_iterations):
		now = dt.datetime.now()
		print now, t1
		elapsed = (now-t1).seconds
		t1 = dt.datetime.now()

		x_batch, y_true_batch, (img_paths,) = random_batch(train_set, len(labels), batch_size=train_batch_size)

		train_input = {x: x_batch, y_true: y_true_batch}
		batch_acc, _ = tr_session.run([accuracy, train_step], feed_dict=train_input)

		print 'Train Perf: %.1f%% (%.1fs)' % (batch_acc * 100, elapsed)

		if (ii % 4 == 0) or (ii == num_iterations - 1):
			eval_x_batch, eval_y_true_batch, (eval_img_paths,) = random_batch(test_set, len(labels), batch_size=plotdim)
			eval_x_batch = x_batch + eval_x_batch
			eval_y_true_batch = y_true_batch + eval_y_true_batch
			score_input = {x: eval_x_batch, y_true: eval_y_true_batch}
			eval_img_paths = img_paths + eval_img_paths

			batch_acc, batch_results = tr_session.run([accuracy, y_guess_index], feed_dict=score_input)

			print len(x_batch), len(batch_results)
			msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%} (%.1fs)"
			print msg.format(ii, batch_acc, elapsed)

			# fig = plt.figure() if fig is None else fig
			# fig.clf()

			# for jj in range(len(eval_y_true_batch)):
			# 	res = batch_results[jj]
			# 	plt.subplot(plotdim + 1, plotdim, jj + 1)
			# 	plt.title(('T' if jj >= train_batch_size else '') + '[%d/%d]: %s' % (int(np.argmax(eval_y_true_batch[jj])), res, labels[res].name))
			# 	plt.imshow(mpimg.imread(eval_img_paths[jj]))
			# fig.canvas.draw()
			# plt.pause(0.05)
