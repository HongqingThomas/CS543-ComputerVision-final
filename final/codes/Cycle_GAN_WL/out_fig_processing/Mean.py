import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('/Users/liuweichensfile/Desktop/01440.jpg')

blur = cv.blur(img,(5,5))


plt.imshow(blur)
plt.xticks([]), plt.yticks([])
plt.show()