import tensorflow as tf
import prettytensor as pt
import numpy as np
from inception_tensors import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def vectorize_id(idnum, labels_len=256):
	vec = np.zeros(labels_len)
	vec[int(idnum) - 1] = 1
	return vec

def random_batch(train_set, num_labels=256, batch_size = 100):
	bucket_size = int(np.floor(batch_size / num_labels))
	batch = []
	for category in train_set:
		idx = list(np.random.choice(len(category), bucket_size, replace=False))

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

	train_set, test_set = split_data(examples)

	for ii in range(num_iterations):
		x_batch, y_true_batch, (img_paths,) = random_batch(train_set, len(labels), batch_size=4)

		train_input = {x: x_batch, y_true: y_true_batch}
		_ = tr_session.run([train_step], feed_dict=train_input)

		if (ii % 4 == 0) or (ii == num_iterations - 1):
			batch_acc, batch_results = tr_session.run([accuracy, y_guess_index], feed_dict=train_input)
			print len(x_batch), len(batch_results)
			msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
			print(msg.format(ii, batch_acc))
			fig = plt.figure()
			for jj in range(min(len(batch_results), 4)):
				res =  batch_results[jj]
				# print res
				plt.subplot(2, 2, jj + 1)
				plt.title('(%d): %s' % (res, labels[res].name))
				plt.imshow(mpimg.imread(img_paths[jj]))
			# plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
			fig.tight_layout()
			plt.show()
			raw_input(':')
