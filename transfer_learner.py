import sys, os

# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import numpy as np

import data_paths
# import cifar10, inception
import food
import tensorflow as tf
from inception_tensors import *
from random import shuffle
from inception_trainer import train


class Transfer:
	def __init__(self):
		self.in_session = tf.Session(graph=in_graph)

if __name__ == '__main__':
	# classify()
	# in_session.run(tf.global_variables_initializer())
	# Create a TensorFlow session for executing the graph.
	# tlearner = Transfer()

	# read labels on food data

	limit_learn_scope = 2
	labels = food.labels()[:limit_learn_scope]
	flattened_examples = []
	for lbl in labels:
		examples = food.examples(lbl, load_images=False)
		flattened_examples += [(one, lbl.name, lbl.id) for one in examples]
	print '1. Loaded food data:', len(flattened_examples), flattened_examples[0]

	guess_vector = classify(flattened_examples[0][0])
	print guess_vector.shape


	shuffle(flattened_examples)
	set_size = len(flattened_examples)
	num_reserved = 2
	train_set, test_set = flattened_examples[:-num_reserved], flattened_examples[-num_reserved:]
	print '2. Split data: ', len(train_set), len(test_set)

	print '# Examples:', transfer_len, len(labels)

	train(train_set, labels)