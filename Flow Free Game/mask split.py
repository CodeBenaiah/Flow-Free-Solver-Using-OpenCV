import cv2
import numpy as np 
from matplotlib import pyplot as plt


path = "/home/ben/Code/Flow-Free-Solver-Using-OpenCV/Flow Free Game/6x6.jpg"
height = 625
width = 500

img = cv2.imread(path)
img = cv2.resize(img,(width, height))
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)

_, th4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
res = cv2.bitwise_not(img, th4)
imgGray1 = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray1,(5,5),1)
imgThreshold = cv2.adaptiveThreshold(imgBlur,255,1,1,11,2)

cv2.imshow("Original", img)
cv2.imshow("4 - THRESH_TOZERO", th4)
cv2.imshow("Result", res)
cv2.imshow("Threshold", imgThreshold)

cv2.waitKey(0)
cv2.destroyAllWindows(0)