from scipy import misc, io
import numpy as np
import matplotlib.pyplot as plot
import os
from PIL import Image

def rgbToGray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.597, 0.114])

images = []
i = 0

basePath = './Images/train'
newWidth = 150
for file in os.listdir(basePath):
    dirPath = basePath + '/' + file
    for imageFile in os.listdir(dirPath):
        img = Image.open(dirPath + '/' + imageFile)
        newHeight = int((newWidth / float(img.size[0])) * float(img.size[1]))
        img = img.resize((100, 100), Image.ANTIALIAS)
        image = np.array(img)
        gray = rgbToGray(image)
        gray = gray.ravel()
        images.append(gray)
        i += 1
        print (i)
    io.savemat(file + '.mat', {'images': images})
    images = []
