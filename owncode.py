from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt



X, y = get_mnist()
W1 = np.random.uniform(-0.5, 0.5, (20, 784))
W2 = np.random.uniform(-0.5, 0.5, (20, 20))
W3 = np.random.uniform(-0.5, 0.5, (10, 20))
b1 = np.zeros((20))
b2 = np.zeros((20))
b3 = np.zeros((10))

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

output = []

runden = 1

#forward prop
for runde in range(runden):
    #for x in X[59990:]:
        xx = 0
        Neurons_h1 = sigmoid(np.dot(W1, X[xx]) + b1)
        print(X[xx].shape)
        print("lost")
        print(Neurons_h1.shape)
        Neurons_h2 = sigmoid(np.dot(W2, Neurons_h1) + b2)
        Neurons_o = np.dot(W3, Neurons_h2) + b3
        output = softmax(Neurons_o)
        xx += 1
        output = softmax(Neurons_o)

#print(output)



# Show results
while True:
    index = int(input("Enter a number (0 - 59999): "))
    img = X[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")
    plt.title("Number: "+str(y[index]) +" Guess: "+ str(output[index]))
    plt.show()
    break