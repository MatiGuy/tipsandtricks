import cv2
import numpy as np
from matplotlib import pyplot as plt

def enhance(img):
    minIntensity = np.iinfo(img.dtype).min
    maxIntensity = np.iinfo(img.dtype).max
    hist, bins = np.histogram(img.flatten(), maxIntensity+1, [minIntensity, maxIntensity+1])
    threshold = 0.35 * 262144 / 200

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
                a = 1
            currentPixel = (img[x, y] - minHist) * ((255 - 0) / (maxHist - minHist)) + 0
            if currentPixel > 255:
                img[x, y] = 255
            elif currentPixel < 0:
                img[x, y] = 0
            else:
                img[x, y] = currentPixel
    '''
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.show()
    '''

    return img

img1 = cv2.imread('lena.jpg', 0)
img2 = cv2.imread('test1.tiff', 0)
img3 = cv2.imread('test2.tiff', 0)

cv2.imwrite("stretchImageLena.jpg", enhance(img1))
cv2.imwrite("stretchTest1.tiff", enhance(img2))
cv2.imwrite("stretchTest2.tiff", enhance(img3))


