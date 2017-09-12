
import data_paths
import os, inception
from inception_namespace import *
import tensorflow as tf

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