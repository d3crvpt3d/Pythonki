import numpy as np
from PIL import Image

X = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

#inputs aus bildern in array mit richtgen guess in stelle "0" (0 = linie von links oben nach rechts unten, 1 = mitte oben unten, 2 = ) 1 und 4 und 5 sind "0"
i = np.random.randint(1, 10)
if i == 1:
    j = 0
if i == 2:
    j = 1
if i == 3:
    j = 9
if i == 4:
    j = 0                       #key
if i == 5:
    j = 0
if i == 6:
    j = 9
if i == 7:
    j = 9
if i == 8:
    j = 9
if i == 9:
    j = 9
image = Image.open("9pix_"+str(i)+"_["+str(j)+"].png")

position = 1

for y in range(3):

    for x in range(3):

        X[position] = image.getpixel((x, y))
        print(X[position])
        position += 1

X[0] = j

#forward prop
#Z1 = W1 * X + b

print(str(i) +" "+ str(j))