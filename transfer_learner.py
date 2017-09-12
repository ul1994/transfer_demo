import sys, os

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

import data_paths
import cifar10, inception
import food
import tensorflow as tf
from inception_tensors import *
from random import shuffle

class Transfer:
	def __init__(self):
		self.in_session = tf.Session(graph=in_graph)

if __name__ == '__main__':
	# Create a TensorFlow session for executing the graph.
	tlearner = Transfer()

	# read labels on food data

	labels = food.labels()
	flattened_examples = []
	for lbl in labels:
		examples = food.examples(lbl, load_images=False)
		flattened_examples += [(one, lbl.name, lbl.id) for one in examples]
	print '1. Loaded food data:', len(flattened_examples), flattened_examples[0]


	shuffle(flattened_examples)
	set_size = len(flattened_examples)
	num_reserved = 2
	train_set, test_set = flattened_examples[:-num_reserved], flattened_examples[-num_reserved:]
	print '2. Split data: ', len(train_set), len(test_set)

	print transfer_len, len(labels)
	x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')
	y_true = tf.placeholder(tf.float32, shape=[None, len(labels)], name='y_true')
	y_correct = tf.argmax(y_true, axis=1)

	import prettytensor as pt
	x_pretty = pt.wrap(x)

	with pt.defaults_scope(activation_fn=tf.nn.relu):
		y_output, loss = x_pretty                                        \
			.fully_connected(size=len(labels), name='PTT_layer_fc1')    \
			.softmax_classifier(num_classes=len(labels), labels=y_true)

	global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
	optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step)
	y_guess = tf.argmax(y_output, axis=1)
	y_is_correct = tf.equal(y_guess, y_correct)
	accuracy = tf.reduce_mean(tf.cast(y_is_correct, tf.float32))
	tr_session = tf.Session()
	print '3. Defined auxillary calcs.'

	train_writer = tf.summary.FileWriter('logs/train', tr_session.graph)
# session.run(tf.global_variables_initializer())
