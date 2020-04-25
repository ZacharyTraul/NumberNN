import numpy as np
import matplotlib.pyplot as plt
import json
import time

class NeuralNetwork:
	"""A neural network, what else would this be?"""
	def __init__(self, network, input_data, output_data, learning_rate):
		"""Creates a neural network with the number of layers and nodes described in the variable network"""
		self.input_data = input_data
		self.output_data = output_data
		self.element = 0
		#Network is of the form: [nodes in input, nodes in hidden layer 1, ..., nodes in hidden layer n, nodes in output layer]
		#Create the weights, layers, and biases based on the input network
		self.num_layers = len(network)
		#Each is a list of numpy arrays
		self.layers = []
		self.weights = []
		self.biases = []
		for i in range(self.num_layers):
			#Layers set to 0
			self.layers.append(np.zeros((network[i])))
			#Weights set randomly between 0 and 1, no weights after the outputs
			if i+1 != self.num_layers:
				self.weights.append(np.random.uniform(-0.1, 0.1, (network[i+1], network[i])))
			#Biases set randomly between 0 and 1, no bias on inputs
			if i != 0:
				self.biases.append(np.random.rand(network[i]))
		#Back vars
		self.errors = []
		self.error = 0
		self.error_d = 0
		self.layers_d = []
		self.layers_e = []
		#Gradients vars
		self.learning_rate = learning_rate
		self.weights_g = []
		self.biases_g = []
	
	#I am aware this is not a sigmoid but I don't really care at this point
	def sig(self, x):
		return np.tanh(x)
	
	def dsig(self, x):
		return 1/np.power(np.cosh(x), 2)
	
	def feed_forward(self, inp):
		self.layers[0] = inp
		self.layers_d = [None]
		
		for i in range(self.num_layers-1):
			total = np.dot(self.weights[i], self.layers[i]) + self.biases[i]
			self.layers[i+1] = self.sig(total)
			self.layers_d.append(self.dsig(total))
	
	def calc_error(self, output):
		self.error = np.sum(0.5 * np.power((self.layers[-1] - output[self.element]), 2))
		self.errors.append(self.error)
		self.error_d = (self.layers[-1] - self.output_data[self.element])
	
	def back_propogate(self):
		self.layers_e = [self.layers_d[-1] * self.error_d]

		for i in range(2, self.num_layers):
			weighted_errors = np.dot(self.weights[-(i-1)].T, self.layers_e[-(i-1)])
			self.layers_e.insert(0, self.layers_d[-i] * weighted_errors)
	
	def calc_gradient(self):
		self.weights_g = []
		self.biases_g = []

		for i in range(self.num_layers-1):
			#Make this better I shouldn't have to make two more arrays when I already have them
			self.weights_g.append(-self.learning_rate * np.dot(np.array([self.layers_e[i]]).T, np.array([self.layers[i]])))
		
		self.biases_g = []
		for i in range(self.num_layers-1):
			self.biases_g.append(self.layers_e[i] * -self.learning_rate)
		
	def update_network(self):
		for i in range(self.num_layers-1):
			self.weights[i] += self.weights_g[i]
			self.biases[i] += self.biases_g[i]

	def train(self):
		pp = 0
		error_avg = []
		
		for i in range(len(self.input_data)):
			self.element = i
			self.feed_forward(self.input_data[self.element])
			self.calc_error(self.output_data)
			self.back_propogate()
			self.calc_gradient()
			self.update_network()
			
			#guess = np.argmax(self.layers[-1])
			#actual = np.argmax(self.output_data[self.element])
			#print(f'{i}| Guess: {guess} Actual: {actual}')
			error_avg.append(sum(self.errors[-100:-1])/100)
			print(f'Training Progress: {i} of {len(self.input_data)} Error: {sum(self.errors[-100:-1])/100}\r', end="")
			if i%1000 == 0:
				plt.plot(self.errors, '.')
				plt.plot(error_avg)
				plt.draw()
				plt.pause(0.001)

	def test(self, test_inputs, test_outputs):		
		successes = 0
		failures = 0
		
		for i in range(len(test_inputs)):
			self.feed_forward(test_inputs[i])
			guess = np.argmax(self.layers[-1])
			#print(self.layers[-1])
			#print(f'Guess: {guess} Actual: {test_outputs[i]}')
			if guess == test_outputs[i]:
				#print("success!")
				successes += 1
			else:
				#print("failure!")
				failures += 1
			print(f'Testing Progress: {i} of {len(test_inputs)}\r', end="")
			
		print(successes/(successes+failures))
	
	def classify_image(self, image):
		self.feed_forward(image)
		return np.argmax(self.layers[-1])
	
	def view_output(self, image):
		self.feed_forward(np.asarray(image))
		image = np.asarray(image).reshape((28, 28))
		
		print(f'Guess: {np.argmax(self.layers[-1])}')
		plt.imshow(image, cmap="Greys")
		plt.show()
		
		
	
	def export_network(self, filename):
		w = []
		for weight in self.weights:
			w.append(weight.tolist())
		
		b = []
		for bias in self.biases:
			b.append(bias.tolist())
		
		network = {}
		network['weights'] = w
		network['biases'] = b

		with open(filename, "w") as export_file:
			json.dump(network, export_file)
	
	def import_network(self, filename):
		self.weights = []
		self.biases = []
		
		with open(filename, "r") as import_file:
			network = json.load(import_file)
			
		for weight in network['weights']:
			self.weights.append(np.asarray(weight))
			
		for bias in network['biases']:
			self.biases.append(np.asarray(bias))

	
	def display_network(self):
		print(f'Layers: {self.layers}')
		print(f'Weights: {self.weights}')
		print(f'Biases: {self.biases}')

	def display_learning(self):
		plt.plot(self.errors)
		plt.show()
	
