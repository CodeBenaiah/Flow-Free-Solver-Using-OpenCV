#!/usr/bin/env python3
import cv2
from math import sqrt
import numpy as np 
from matplotlib import pyplot as plt
import sys, string, os

#########################################
imgdata = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
imgdata[14] = cv2.imread("Dataset/15.png")
imgdata[13] = cv2.imread("Dataset/14.png")
imgdata[12] = cv2.imread("Dataset/13.png")
imgdata[11] = cv2.imread("Dataset/12.png")
imgdata[10] = cv2.imread("Dataset/11.png")
imgdata[9] = cv2.imread("Dataset/10.png")
imgdata[8] = cv2.imread("Dataset/9.png")
imgdata[7] = cv2.imread("Dataset/8.png")
imgdata[6] = cv2.imread("Dataset/7.png")
imgdata[5] = cv2.imread("Dataset/6.png")
imgdata[4] = cv2.imread("Dataset/5.png")
imgdata[3] = cv2.imread("Dataset/4.png")
imgdata[2] = cv2.imread("Dataset/3.png")
imgdata[1] = cv2.imread("Dataset/2.png")
imgdata[0] = cv2.imread("Dataset/1.png")

#########################################
def getContours(img, original_img):
    contours,hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    biggest = np.array([])
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>60000:
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt, 0.002*peri,True)
            cv2.drawContours(original_img,cnt,-1,(0,255,0),3)
            if area>max_area and len(approx)==4:
                max_area = area
                biggest = approx
    # cv2.imshow("OG", original_img)
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

def splitBoxes(img,n):
    rows = np.vsplit(img,n)
    boxes=[]
    for r in rows:
        cols = np.hsplit(r,n)
        for box in cols:
            boxes.append(box)
    global box_dimension 
    box_dimension = boxes[0].shape
    print(box_dimension)
    return boxes

col_val = [[0,0,254,1],[0,141,2,2],[4,224,234,3],[254,41,14,4],[0,137,250,5],[35,51,234,6],[243,40,19,7],[74,224,231,8],[38,139,61,9]]

def predict(boxes):
    board = []
    for image in boxes:
        col = (image[50,50])
        print(col)
        e = 0
        for i in col_val:
            if i[0]==col[0] and i[1]==col[1] and i[2]==col[2]:
                e = i[3]
        board.append(e)
    return board

def display_output(img, output,n):
    a = b = 0
    k = l = 100
    m=0
    for i in range(0,n):
        for j in range(0,n):
            cv2.rectangle(img, pt1=(a,b),pt2=(k,l),color=col_val[int(output[m])-1],thickness=-1)
            a +=100
            k += 100
            m+=1
        a = 0
        k = 100
        b+=100
        l+=100
    cv2.imshow("Final", imgWarpColored)

#########################################
height = 500
width = 500
path = "/home/ben/Code/Flow-Free-Solver-Using-OpenCV/Flow Free Game/5x5.jpg"

img = cv2.imread(path)
img = cv2.resize(img,(width, height))
imgBlank = np.zeros((height,width, 3), np.uint)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
_, th4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
res = cv2.bitwise_xor(img, th4)
imgCanny = cv2.Canny(res,100,200)
img_copy = img.copy()

biggest, maxArea = getContours(imgCanny,img_copy)
print(maxArea)
if biggest.size!= 0:
    biggest = reorder(biggest)
    print(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    cv2.drawContours(img_copy, biggest, -1,(0,255,0),10)
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (width,height))
    imgDetectedCircles = imgBlank.copy()

imgWarpColored = cv2.imread("5x5 Warped.png")
imgSolvedDigits = imgBlank.copy()
n = 5
boxes = splitBoxes(imgWarpColored,n)
board = predict(boxes)
bt  = open('board.txt','w')
bt.write(str(n)+"\n")
for i in range(len(board)):
    bt.write(str(board[i])+" ")
bt.close()

os.system("./a.out")

ot = open('output.txt','r')
data = ot.read()
output = data.split(" ")
print(output)

cv2.imshow("Original", imgWarpColored)
display_output(imgWarpColored,output,n)

cv2.waitKey(0)
cv2.destroyAllWindows()