from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt



X, y = get_mnist()
W1 = np.random.uniform(-0.5, 0.5, (784, 20))
W2 = np.random.uniform(-0.5, 0.5, (20, 20))
W3 = np.random.uniform(-0.5, 0.5, (20, 10))

b1 = np.zeros((20, 1))
b2 = np.zeros((20, 1))
b3 = np.zeros((10, 1))

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def sigmoid_derivative(x):
    sd = sigmoid(x)*(1-sigmoid(x))
    return sd

def softmax(vector):
	e = np.exp(vector)
	return e / e.sum()

def ReLU(x):
    if x < 0:
        return 0
    else:
        return x

Neurons_h1 = [20]
Neurons_h2 = [20]
Neurons_o = [10]
output = [10]
runden = 1

#forward prop
for runde in range(runden):
    for x in X:
        for pixel in range(784):
            for xx in range(20):
                Neurons_h1[xx] = sigmoid(np.dot(X[runde], W1.T) + b1[xx])
            for xx in range(20):
                Neurons_h2[xx] = sigmoid(np.dot(Neurons_h1, W2.T) +b2[xx])
            Neurons_o = np.dot(Neurons_h2, W3[pixel]) + b3
            output = softmax(Neurons_o)
            pixel += 1



# Show results
while True:
    index = int(input("Enter a number (0 - 59999): "))
    img = X[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")
    plt.title("Number: "+str(y[index]) +" Guess: "+ str(output[index]))
    plt.show()