from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt


X, y = get_mnist() #X[0-5999 pictures, 0-783 pixel in each picture] y[0-59999 pictures, 0-9 real results in array]
W1 = np.random.uniform(low = 0, high = 1, size = (20, 784))
W2 = np.random.uniform(low = 0, high = 1, size = (20, 20))
W3 = np.random.uniform(low = 0, high = 1, size = (10, 20))
b1 = np.zeros((20))
b2 = np.zeros((20))
b3 = np.zeros((10))


#activation functions
def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def sigmoid_derivative(x):
    sd = sigmoid(x)*(1-sigmoid(x))
    return sd

def softmax(x):
	e = np.exp(x)
	return e / e.sum()

def ReLU(x):
    if x < 0:
        return 0
    else:
        return x


#some global variables
output = []
runden = 1
update = 0

thisoff = 0
off = 10 #fehlerquote = max

W1_best = W1
W2_best = W2
W3_best = W3

W1_test = W1
W2_test = W2
W3_test = W3

for runde in range(runden):
    xx = 0
    for x in X[:10]:
        
        for i in range(10):
            #randomness based on how off the current of lvl is
            #(test)
            #every weight updaten
            for f in range(len(W1)):
                for q in range(len(W1[0])):
                    W1_test[f][q] = 0 - W1_best[f][q] + np.random.random() * 0.5
            for f in range(len(W2)):
                for q in range(len(W2[0])):
                    W2_test[f][q] = 0 - W2_best[f][q] + np.random.random() * 0.5
            for f in range(len(W3)):
                for q in range(len(W3[0])):
                    W2_test[f][q] = 0 - W3_best[f][q] + np.random.random() * 0.5


            #forward prop
            Neurons_h1 = sigmoid(np.dot(W1_test, X[xx]) + b1)
            Neurons_h2 = sigmoid(np.dot(W2_test, Neurons_h1) + b2)
            Neurons_o = sigmoid(np.dot(W3_test, Neurons_h2) + b3)
            output = softmax(Neurons_o)

            print("output")
            print(output)
            print()

            #backprop (not best cuz i dont want to do this real yet)

            #calculate off
            thisoff = sum( abs(output - y[xx]) )
            print("output")
            print(output)
            print("y")
            print(y[xx])


            #save every weight and bias if output is better then before
            if thisoff < off:
                off = thisoff

                W1_best
                W2_best
                W3_best
            
            '''
            #only if something is wrong
            b1_best = b1
            b2_best = b2
            b3_best = b3
            '''

            #update weights on the best out of every few nn's
            print("Start "+str(xx)+" update...")
            
            print("Updated! Current error: "+str(off))
            

        #count for X
        xx += 1


# Show results

index = int(input("Enter a number (0 - 59999): "))
img = X[index]
plt.imshow(img.reshape(28, 28), cmap="Greys")
plt.title("Number: "+str(y[index].argmax()) +" Guess: "+ str(output.argmax()))
plt.show()