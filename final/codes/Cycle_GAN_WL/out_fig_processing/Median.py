import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('/Users/liuweichensfile/Desktop/01440.jpg')

median = cv.medianBlur(img,5)

plt.imshow(median)
plt.xticks([]), plt.yticks([])
plt.show()