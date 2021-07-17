import numpy as np
from PIL import Image

X = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
X1 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

#inputs aus bildern in array mit richtgen guess in stelle "0" (0 = linie von links oben nach rechts unten, 1 = mitte oben unten, 2 = ) 1 und 4 und 5 sind "0"
def training(Name_output_model, W1, W2, W3):
    i = np.random.randint(1, 10)
    if i == 1:
        j = 0
    elif i == 2:
        j = 1
    elif i == 3:
        j = 9
    elif i == 4:
        j = 0                       #key
    elif i == 5:
        j = 0
    elif i == 6:
        j = 9
    elif i == 7:
        j = 9
    elif i == 8:
        j = 9
    elif i == 9:
        j = 9

    image = Image.open("9pix_"+str(i)+"_["+str(j)+"].png")

    position = 1

    for y in range(3): #fill pixelvalues to array X (X[0] = right guess)

        for x in range(3):

            tmp = image.getpixel((x, y))
            X[position] = tmp[0]
            position += 1

    X[0] = j #first position is right guess


    print("I: "+str(i)+" J: "+str(j))
    for f in range(10):
        print("X"+str(f)+": "+str(X[f]))

    for i in range(9):
        X1[i] = X[i+1]
    
    #speichern
    f = open(str(Name_output_model)+".KI_model_W1", "w")
    xx = 0
    for x in W1:
        f.write(str(W1[xx]))
    
    f = open(str(Name_output_model)+".KI_model_W2", "w")
    xx = 0
    for x in W2:
        f.write(str(W2[xx]))
    
    f = open(str(Name_output_model)+".KI_model_W3", "w")
    xx = 0
    for x in W3:
        f.write(str(W3[xx]))






X1_length = len(X1) #length of X1



def Sample(i, Name_input_model):                                  #use own picture
    image = Image.open("test_"+str(i)+".png")
    xx = 0
    f = open(str(Name_input_model)+".KI_model_W1", "r")
    f.read() ##############
    f.close
    xx = 0
    f = open(str(Name_input_model)+".KI_model_W2", "r")
    f.read() ##############
    f.close
    xx = 0
    f = open(str(Name_input_model)+".KI_model_W3", "r")
    f.read() ##############
    f.close



#init weights and biases

def weights1(X1_length, neuron_anzahl):
    return np.random.randn(X1_length, neuron_anzahl)

def weights2(inputs_anzahl, neuron_anzahl):
    return np.random.randn(inputs_anzahl, neuron_anzahl)

def weights3(inputs_anzahl, neuron_anzahl):
    return np.random.randn(inputs_anzahl, neuron_anzahl)


def bias1(neuron_anzahl):
    return np.zeros((1, neuron_anzahl))
def bias2(neuron_anzahl):
    return np.zeros((1, neuron_anzahl))
def bias3(neuron_anzahl):
    return np.zeros((1, neuron_anzahl))



class ReLU:
    def forward(self, input):
        return np.maximum(0, input)



class Softmax:
    def forward(self, input):
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities


#User Interface
W1 = weights1(9, 10)
W2 = weights2(10, 10)
W3 = weights3(10, 10)

b1 = bias1(10)
b2 = bias2(10)
b3 = bias3(10)

training("Model1", W1, W2, W3) #wie das model heißen soll

Sample(1, "Model1")  #welches bild und welches modell genommen werden soll