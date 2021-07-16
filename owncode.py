import numpy as np
from PIL import Image



for x in range(5):
    image = Image.open("creeper64_"+str(x)+".png")
    image.show()
