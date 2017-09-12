# Bring your packages onto the path
import sys, os

sys.path.append(os.path.abspath(os.environ['CIFAR_PATH']))
sys.path.append(os.path.abspath(os.environ['INCEPTION_PATH']))

import cifar10, inception
from inception import transfer_values_cache
# from cifar10 import num_classes

# class_names = cifar10.load_class_names()

# print cifar10.num_classes, class_names

images_train, cls_train, labels_train = cifar10.load_training_data()

model = inception.Inception()

train_cache = os.path.join(cifar10.data_path, 'train_cache/')
test_cache = os.path.join(cifar10.data_path, 'test_cache/')

images_scaled = images_train * 255.0
transfer_values_train = transfer_values_cache(train_cache,
	images=images_scaled,
	model=model)
