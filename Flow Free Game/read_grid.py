import cv2
import numpy as np
from matplotlib import pyplot as plt

def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur,255,1,1,11,2)
    return imgThreshold

def biggestContours(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area>50:
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            if area>max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area

def reorder(points):
    points = points.reshape((4,2))
    pointsnew = np.zeros((4,1,2),dtype= np.int32)
    add = points.sum(1)
    pointsnew[0] = points[np.argmin(add)]
    pointsnew[3] = points[np.argmax(add)]
    diff = np.diff(points,axis=1)
    pointsnew[1] = points[np.argmin(diff)]
    pointsnew[2] = points[np.argmax(diff)]
    return pointsnew

path = "Flow Free Game/5x5.jpg"
height = 625
width = 500

img = cv2.imread(path)
img = cv2.resize(img,(width, height))
imgBlank = np.zeros((height,width, 3), np.uint)
imgThreshold = preProcess(img)

imgContours = img.copy()
imgBigcontours = img.copy()
contours, heirarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours,contours, -1,(0,255,0),3)

biggest, maxArea = biggestContours(contours)

if biggest.size!= 0:
    biggest = reorder(biggest)
    cv2.drawContours(imgBigcontours, biggest, -1,(0,0,255),10)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (width,height))
    imgDetectedCircles = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

cv2.imshow("Grid", img)
cv2.imshow("Threshold", imgThreshold)
cv2.imshow("Contours", imgContours)
cv2.imshow("Grid Pts", imgBigcontours)

cv2.waitKey(0)
cv2.destroyAllWindows()  