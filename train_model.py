# coding=utf-8

import os
import csv
import time
import random
import operator
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data as mnist_input_data

def main():

	cnn = ConvolutionalNeuralNetwork([
		ConvolutionLayer(5, 1, 64),
		PoolingLayer(2),
		ConvolutionLayer(5, 1, 128),
		PoolingLayer(2),
		FullyConnectedLayer(128),
		FullyConnectedLayer(64),
		FullyConnectedLayer(32),
		DropoutLayer()
	], KatakanaDataSet())
	cnn.build_graph()
	cnn.train_model(steps=200,
					training_batch_size=128,
					evaluation_batch_size=50,
					file_path="model.pb")

# Method definitions

def relative_path(path):
	return os.path.dirname(os.path.realpath(__file__)) + '/' + path

def max_index(array):
	index, value = max(enumerate(array), key=operator.itemgetter(1))
	return index

class ConvolutionalNeuralNetwork:

	def __init__(self, layers, dataset):
		self.layers = layers
		self.dataset = dataset
		layers.append(ReadoutLayer())

	def build_graph(self):

		input_shape = self.dataset.get_input_shape()
		output_size = self.dataset.get_output_size()

		readout = self.layers[-1]
		readout.output_size = output_size

		self.x = tf.placeholder(tf.float32, [None] + input_shape, 'x')
		self.y_truth = tf.placeholder(tf.float32, [None, output_size], 'y_truth')

		tensor = self.x

		for layer in self.layers:
			tensor = layer.build_training_node(tensor)

		self.y = tensor

		self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_truth, logits=self.y)
		self.loss = tf.reduce_mean(self.cross_entropy)

		# TODO adapt learning rate each step to improve learning speed
		self.train_step = tf.train.AdamOptimizer(0.005).minimize(self.loss)
		self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_truth, 1)), tf.float32))

		# Prepare TensorBoard
		tf.summary.image('inputs', self.x, 10)
		tf.summary.scalar('loss', self.loss)
		tf.summary.scalar('accuracy', self.accuracy)

		self.summary = tf.summary.merge_all()

	def train_model(self, steps=500, training_batch_size=None, evaluation_batch_size=None, file_path=False):

		with tf.Session() as session:

			summary_path = relative_path('data/training_summaries/run_{}'.format(str(int(time.time()))))
			summary_writer = tf.summary.FileWriter(summary_path, session.graph)

			session.run(tf.global_variables_initializer())
			for step in range(steps):

				# Train
				x, y_truth = self.dataset.get_training_batch(training_batch_size)
				self.train_step.run(feed_dict=self._feed(0.5, {self.x: x, self.y_truth: y_truth}))

				# Send data to TensorBoard
				x, y_truth = self.dataset.get_test_batch(evaluation_batch_size)
				summary_run = session.run(self.summary, feed_dict=self._feed(1, {self.x: x, self.y_truth: y_truth}))
				summary_writer.add_summary(summary_run, step)

			if file_path:
				self._save_graph(session, file_path)

	def _save_graph(self, session, file_path):

		training_graph = tf.get_default_graph()
		prediction_graph = tf.Graph()

		with prediction_graph.as_default():

			input_shape = self.dataset.get_input_shape()
			output_size = self.dataset.get_output_size()

			readout = self.layers[-1]
			readout.output_size = output_size

			input_placeholder = tf.placeholder(tf.float32, [None] + input_shape, 'input')

			tensor = input_placeholder

			for layer in self.layers:
				if not isinstance(layer, DropoutLayer):
					tensor = layer.build_prediction_node(tensor, session)

			tf.identity(tensor, 'prediction')

			tf.train.write_graph(prediction_graph, os.path.dirname(file_path), os.path.basename(file_path), as_text=False)

	def _feed(self, keep_probability, feed):
		for layer in self.layers:
			if isinstance(layer, DropoutLayer):
				feed[layer.placeholder] = keep_probability

		return feed

class CNNLayer:

	def weight_variables(self, shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def bias_variables(self, shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	def build_training_node(self, input_tensor):
		pass

	def build_prediction_node(self, input_tensor, session):
		pass

class ConvolutionLayer(CNNLayer):

	def __init__(self, patch, stride, features):
		self.patch = patch
		self.stride = stride
		self.features = features
		self._weights = False
		self._biases = False

	def build_training_node(self, input_tensor):

		if not self._weights and not self._biases:
			input_size = input_tensor.shape[-1].value
			self._weights = self.weight_variables([self.patch, self.patch, input_size, self.features])
			self._biases = self.bias_variables([self.features])

		return self._build_node(input_tensor, self._weights, self._biases)

	def build_prediction_node(self, input_tensor, session):
		return self._build_node(input_tensor,
								tf.constant(session.run(self._weights)),
								tf.constant(session.run(self._biases)))

	def _conv2d(self, input_tensor, weights):
		return tf.nn.conv2d(input_tensor, weights,
							strides=[self.stride, self.stride, self.stride, self.stride],
							padding='SAME')

	def _build_node(self, input_tensor, weights, biases):
		return tf.nn.relu(self._conv2d(input_tensor, weights) + biases)

class PoolingLayer(CNNLayer):

	def __init__(self, pooling):
		self.pooling = pooling

	def build_training_node(self, input_tensor):
		return self._build_node(input_tensor)

	def build_prediction_node(self, input_tensor, session):
		return self._build_node(input_tensor)

	def _build_node(self, input_tensor):
		return tf.nn.max_pool(input_tensor,
								ksize=[1, self.pooling, self.pooling, 1],
								strides=[1, self.pooling, self.pooling, 1],
								padding='SAME')

class FullyConnectedLayer(CNNLayer):

	def __init__(self, neurons):
		self.neurons = neurons
		self._weights = False
		self._biases = False

	def build_training_node(self, input_tensor):

		flattened_size = self._flattened_size(input_tensor)

		if not self._weights and not self._biases:
			self._weights = self.weight_variables([flattened_size, self.neurons])
			self._biases = self.bias_variables([self.neurons])

		return self._build_node(input_tensor, self._weights, self._biases)

	def build_prediction_node(self, input_tensor, session):
		return self._build_node(input_tensor,
								tf.constant(session.run(self._weights)),
								tf.constant(session.run(self._biases)))

	def _flattened_size(self, input_tensor):
		return reduce(lambda a, b: a*b, input_tensor.shape[1:]).value

	def _build_node(self, input_tensor, weights, biases):
		flattened_size = self._flattened_size(input_tensor)
		flattened_tensor = tf.reshape(input_tensor, [-1, flattened_size])
		return tf.nn.relu(tf.matmul(flattened_tensor, weights) + biases)

class DropoutLayer(CNNLayer):

	def __init__(self):
		self.placeholder = tf.placeholder(tf.float32, name='dropout_probability')

	def build_training_node(self, input_tensor):
		return self._build_node(input_tensor)

	def build_prediction_node(self, input_tensor, session):
		return self._build_node(input_tensor)

	def _build_node(self, input_tensor):
		return tf.nn.dropout(input_tensor, self.placeholder)

class ReadoutLayer(CNNLayer):

	def __init__(self, output_size=None):
		self.output_size = output_size
		self._weights = False
		self._biases = False

	def build_training_node(self, input_tensor):

		if not self._weights and not self._biases:
			input_size = input_tensor.shape[-1].value
			self._weights = self.weight_variables([input_size, self.output_size])
			self._biases = self.bias_variables([self.output_size])

		return self._build_node(input_tensor, self._weights, self._biases)

	def build_prediction_node(self, input_tensor, session):
		return self._build_node(input_tensor,
								tf.constant(session.run(self._weights)),
								tf.constant(session.run(self._biases)))

	def _build_node(self, input_tensor, weights, biases):
		return tf.matmul(input_tensor, weights) + biases

class DataSet():

	def get_input_shape(self):
		pass

	def get_output_size(self):
		pass

	def get_validation_batch(self, size=None):
		pass

	def get_training_batch(self, size=None):
		pass

	def get_test_batch(self, size=None):
		pass

class MNISTDataSet(DataSet):

	def __init__(self):
		self._mnist = mnist_input_data.read_data_sets("data/MNIST_data/", one_hot=True)

	def get_input_shape(self):
		return [28, 28, 1]

	def get_output_size(self):
		return 10

	def get_validation_batch(self, size=None):
		return self._get_batch(self._mnist.validation, size, 5000)

	def get_training_batch(self, size=None):
		return self._get_batch(self._mnist.train, size, 55000)

	def get_test_batch(self, size=None):
		return self._get_batch(self._mnist.test, size, 10000)

	def _get_batch(self, data, size, default_size):
		if size == None:
			size = default_size
		batch = data.next_batch(size)
		inputs = batch[0].reshape((-1, 28, 28, 1))
		classification = batch[1]
		return inputs, classification

class KatakanaDataSet(DataSet):

	def __init__(self):
		self.categories = []
		self.categories_display = {}

		with open(relative_path('data/katakana/categories.csv')) as file:
			reader = csv.reader(file)
			reader.next()
			for category, display in reader:
				self.categories_display[int(category)] = display
				self.categories.append(int(category))

		self.classification = []
		with open(relative_path('data/katakana/classification.csv')) as file:
			reader = csv.reader(file)
			reader.next()
			for position, category in reader:
				self.classification.append((int(position), int(category)))

	def get_input_shape(self):
		return [64, 64, 1]

	def get_output_size(self):
		return len(self.categories)

	def get_validation_batch(self, size=None):
		if size == None:
			size = len(self.classification)*0.2
		return self._get_batch(0, len(self.classification)*0.2, size)

	def get_training_batch(self, size=None):
		if size == None:
			size = len(self.classification)*0.6
		return self._get_batch(len(self.classification)*0.2, len(self.classification)*0.8, size)

	def get_test_batch(self, size=None):
		if size == None:
			size = len(self.classification)*0.2
		return self._get_batch(len(self.classification)*0.8, len(self.classification), size)

	def _get_batch(self, start, end, length):
		inputs = []
		classification = []
		categories_size = len(self.categories)
		with open(relative_path('data/katakana/data')) as data_file:
			for i in random.sample(range(int(start), int(end)), length):
				position, category = self.classification[i]
				inputs.append(self._image_data(data_file, position))
				classification.append(self._one_hot(self.categories.index(category), categories_size))

		return inputs, classification

	def _image_data(self, file, position):
		file.seek(position * 512)
		data = np.unpackbits(np.frombuffer(file.read(512), dtype=np.uint8))
		data = data.reshape([64, 64, 1])
		return data

	def _one_hot(self, index, length):
		vector = np.zeros(length)
		vector[index] = 1
		return vector

class Classifier:

	def __init__(self, file_path):
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(open(file_path, 'rb').read())
		tf.import_graph_def(graph_def, name='')

		graph = tf.get_default_graph()

		self.input_placeholder = graph.get_tensor_by_name('input:0')
		self.prediction = graph.get_tensor_by_name('prediction:0')

	def classify(self, input):
		with tf.Session() as session:
			return session.run(self.prediction, feed_dict={ self.input_placeholder: input})

# Runtime

if __name__ == '__main__':
	main()