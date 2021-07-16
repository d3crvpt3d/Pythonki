import PIL
import numpy as np
from PIL import Image , ImageOps
import matplotlib.pyplot as plt





for i in range(4):
    if i == 0:
        i = 1
    print("Picture Number: "+ str(i))
    og_image = Image.open("creeper ("+str(i)+").png")
    for x in range(64):
        for y in range(64):
            pixel = og_image.getpixel((x, y))
            print(pixel)

    



'''
for i in range(5):
    i += 1
    og_image = Image.open("notcreeper ("+str(i)+").jpg")
'''

