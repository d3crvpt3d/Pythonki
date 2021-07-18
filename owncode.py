from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt



X, y = get_mnist()
W1 = np.random.uniform(-0.5, 0.5, (20, 784))
W2 = np.random.uniform(-0.5, 0.5, (20, 20))
W3 = np.random.uniform(-0.5, 0.5, (10, 20))
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
        test = 0
        for xx in range(20):
            print(xx)
            Neurons_h1[xx] = sigmoid(W1[xx] @ X[test]) + b1[xx]
        for xx in range(20):
            Neurons_h2[xx] = sigmoid(W2[xx] @ Neurons_h1[xx]) + b2[xx]
        for xx in range(10):
            Neurons_o[xx] = W3[xx] @ Neurons_h2[xx].T + b3
        test += 1
    output = softmax(Neurons_o)



# Show results
while True:
    index = int(input("Enter a number (0 - 59999): "))
    img = X[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")
    plt.title("Number: "+str(y[index]) +" Guess: "+ str(output[index]))
    plt.show()