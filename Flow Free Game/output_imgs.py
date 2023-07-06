import cv2

col_val = [[0,0,254,1],[0,141,2,2],[4,224,234,3],[254,41,14,4],[0,137,250,5]]

path = "/home/ben/Code/Flow-Free-Solver-Using-OpenCV/Flow Free Game/5x5.jpg"

a = b = 25
k = l = 75
m=0
for i in range(0,n):
    for j in range(0,n):
        cv2.rectangle(img, pt1=(a,b),pt2=(k,l),color=col_val[output[m]-1],thickness=-1)
        a +=100
        k += 100
        m+=1
        print(k,l)
    a = 25
    k = 75
    b+=100
    l+=100

cv2.imshow("sfdl", img)
cv2.waitKey(0)
cv2.destroyAllWindows()