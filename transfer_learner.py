import sys, os

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

import data_paths
import cifar10, inception
from food import food_labels, food_examples
import tensorflow as tf
from inception_namespace import *
from inception_tensors import *

class Transfer:
	def __init__(self):
		in_session = tf.Session(graph=in_graph)

if __name__ == '__main__':
	# Create a TensorFlow session for executing the graph.
	tlearner = Transfer()
