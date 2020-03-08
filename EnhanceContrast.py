import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lena.jpg', 0)
img = cv2.imread('test1.tiff', 0)
img = cv2.imread('test2.tiff', 0)

hist,bins = np.histogram(img.flatten(),256,[0,256])
threshold = 0.35*262144/200

minCounter = 0
maxCounter = 0
minHist = 0
maxHist = 0

for i in range(hist.size):
    minCounter += hist[i]
    if minCounter >= threshold:
        minHist = i
        break
for i in reversed(range(hist.size)):
    maxCounter += hist[i]
    if maxCounter >= threshold:
        maxHist = i
        break

rows = img.shape[0]
cols = img.shape[1]
for x in range(0, rows):
    for y in range(0, cols):
        if x == 324 and y == 321:
            a=1
        currentPixel = (img[x, y] - minHist) * ((255 - 0) / (maxHist - minHist)) + 0
        if currentPixel > 255:
            img[x, y] = 255
        elif currentPixel < 0:
            img[x, y] = 0
        else:
            img[x, y] = currentPixel



# cv2.imwrite("stretchImageLena.jpg", img)
# cv2.imwrite("stretchTest1.tiff", img)
# cv2.imwrite("stretchTest2.tiff", img)

'''
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.show()
'''
