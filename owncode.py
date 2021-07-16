import numpy as np
from PIL import Image



for x in range(5):
    image = Image.open("creeper64_"+str(x)+".png")
    image.show()
    KI.
    

class KI:
    
    def 
    
    X = []
    X.append()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases