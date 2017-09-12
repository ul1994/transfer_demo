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
