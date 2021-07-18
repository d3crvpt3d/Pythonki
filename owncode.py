from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt



X, y = get_mnist()
W1 = np.random.uniform(-0.5, 0.5, (784, 1))
W2 = np.random.uniform(-0.5, 0.5, (20, 1))
W3 = np.random.uniform(-0.5, 0.5, (20, 1))

b1 = np.zeros((784, 1))
b2 = np.zeros((20, 1))
b3 = np.zeros((20, 1))

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def sigmoid_derivative(x):
    sd = sigmoid(x)*(1-sigmoid(x))
    return sd

Neurons_h1 = [20]
Neurons_h2 = [20]
Neurons_o = [10]

#forward prop
for x in X:
        Neurons_h1 = np.dot(X[x], W1.T)


# Show results
while True:
    index = int(input("Enter a number (0 - 59999): "))
    img = X[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")
    plt.title(str(y[index]) + str(Neurons_o[index]))
    plt.show()