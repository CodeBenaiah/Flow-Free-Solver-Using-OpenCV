import cv2
from math import sqrt
import numpy as np 
from matplotlib import pyplot as plt

#########################################
imgdata = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
imgdata[14] = cv2.imread("Flow Free Game/Dataset/15.png")
imgdata[13] = cv2.imread("Flow Free Game/Dataset/14.png")
imgdata[12] = cv2.imread("Flow Free Game/Dataset/13.png")
imgdata[11] = cv2.imread("Flow Free Game/Dataset/12.png")
imgdata[10] = cv2.imread("Flow Free Game/Dataset/11.png")
imgdata[9] = cv2.imread("Flow Free Game/Dataset/10.png")
imgdata[8] = cv2.imread("Flow Free Game/Dataset/9.png")
imgdata[7] = cv2.imread("Flow Free Game/Dataset/8.png")
imgdata[6] = cv2.imread("Flow Free Game/Dataset/7.png")
imgdata[5] = cv2.imread("Flow Free Game/Dataset/6.png")
imgdata[4] = cv2.imread("Flow Free Game/Dataset/5.png")
imgdata[3] = cv2.imread("Flow Free Game/Dataset/4.png")
imgdata[2] = cv2.imread("Flow Free Game/Dataset/3.png")
imgdata[1] = cv2.imread("Flow Free Game/Dataset/2.png")
imgdata[0] = cv2.imread("Flow Free Game/Dataset/1.png")

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

col_val = [[4,249,15,1],[42,40,170,2],[0,0,254,3],[255,255,255,4],[255,255,4,5],[200,8,255,6],[82,139,161,7],[187,159,159,8],[177,41,57,9],[254,41,14,10],[127,0,129,11],[0,137,250,12],[0,141,0,13],[4,224,234,14],[129,128,0,15]]

def predict(boxes):
    board = []
    for image in boxes:
        col = (image[50,50])
        e = 0
        for i in col_val:
            if i[0]==col[0] and i[1]==col[1] and i[2]==col[2]:
                e = i[3]
        board.append(e)
    return board

def display_output(img, output,n):
    secw = int(img.shape[1]/n)
    sech = int(img.shape[0]/n)
    i = 0
    x_offset=y_offset=50
    for x in range(0,n):
        for y in range(0,n):
            img[y_offset:y_offset+100, x_offset:x_offset+100] = imgdata[output[i]]



#########################################
height = 500
width = 500
path = "/home/ben/Code/Flow-Free-Solver-Using-OpenCV/Flow Free Game/5x5.jpg"

img = cv2.imread(path)
img = cv2.resize(img,(width, height))
imgBlank = np.zeros((height,width, 3), np.uint)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(3,3),0)
_, th4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
res = cv2.bitwise_xor(img, th4)
imgCanny = cv2.Canny(res,100,200)
img_copy = img.copy()

biggest, maxArea = getContours(imgCanny,img_copy)
print(maxArea)
if biggest.size!= 0:
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    cv2.drawContours(img_copy, biggest, -1,(0,255,0),10)
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (width,height))
    imgDetectedCircles = imgBlank.copy()

imgSolvedDigits = imgBlank.copy()
n = 5
boxes = splitBoxes(imgWarpColored,n)
board = predict(boxes)
print(board)
bt  = open('board.txt','w')
bt.write(str(n)+"\n")
for i in range(len(board)):
    bt.write(str(board[i])+" ")
bt.close()


cv2.imshow("Original", imgWarpColored)

cv2.waitKey(0)
cv2.destroyAllWindows()