import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('/Users/liuweichensfile/Desktop/01440.jpg')


blur = cv.GaussianBlur(img,(5,5),0)


plt.imshow(blur)#,plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()