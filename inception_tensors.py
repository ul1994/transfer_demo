
import data_paths
import os, inception
from inception_namespace import *
import tensorflow as tf
import numpy as np
import matplotlib as plt

path_graph_def = "classify_image_graph_def.pb"
model_path = os.path.join(inception.data_dir, path_graph_def)
print 'Locationg model at:', model_path

in_graph = tf.Graph()
with in_graph.as_default():
	with tf.gfile.FastGFile(model_path, 'rb') as file:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(file.read())
		tf.import_graph_def(graph_def, name='')

# Get the output of the Inception model by looking up the tensor
# with the appropriate name for the output of the softmax-classifier.
y_pred = in_graph.get_tensor_by_name(tensor_name_softmax)

# Get the unscaled outputs for the Inception model (aka. softmax-logits).
y_logits = in_graph.get_tensor_by_name(tensor_name_softmax_logits)

# Get the tensor for the resized image that is input to the neural network.
resized_image = in_graph.get_tensor_by_name(tensor_name_resized_image)

# Get the tensor for the last layer of the graph, aka. the transfer-layer.
transfer_layer = in_graph.get_tensor_by_name(tensor_name_transfer_layer)

# Get the number of elements in the transfer-layer.
transfer_len = transfer_layer.get_shape()[3]

in_session = tf.Session(graph=in_graph)


def tr_create_feed_dict(image_path):
	image_data = tf.gfile.FastGFile(image_path, 'rb').read()
	feed_dict = {tensor_name_input_jpeg: image_data}
	return feed_dict

def classify(image_path):
	feed_dict = tr_create_feed_dict(image_path)
	pred = in_session.run(transfer_layer, feed_dict=feed_dict)
	pred = np.squeeze(pred)
	return pred

def aspatch(flatarr):
	ln = len(flatarr)
	width = int(np.ceil(np.sqrt(ln)))
	patch = np.zeros((width, width))
	for yy in range(width):
		for xx in range(width):
			flatindex = yy * width + xx
			if flatindex < ln:
				patch[yy][xx] = flatarr[flatindex]
	plt.imshow(aspatch(patch), interpolation='nearest', cmap='Reds')
	plt.show()
