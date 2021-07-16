import PIL
import numpy as np
from PIL import Image , ImageOps



for i in range(5):
    if i == 1:
        i =- 1
    print("Picture Number: "+ str(i))
    og_image = Image.open("creeper ("+str(i)+").png")
    ImageOps.grayscale(og_image)
    for x in range(64):
        if x == 1:
            x =- 1
        for y in range(64):
            if y == 1:
                y =- 1
            pixel = og_image.getpixel((x, y))
            print(pixel)

    



'''
for i in range(5):
    i += 1
    og_image = Image.open("notcreeper ("+str(i)+").jpg")
'''

