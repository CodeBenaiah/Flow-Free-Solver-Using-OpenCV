import cv2

col_val = [[4,249,15,1],[42,40,170,2],[0,0,254,3],[255,255,255,4],[255,255,4,5],[200,8,255,6],[82,139,161,7],[187,159,159,8],[177,41,57,9],[254,41,14,10],[127,0,129,11],[0,137,250,12],[0,141,0,13],[4,224,234,14],[129,128,0,15]]

path = "/home/ben/Code/Flow-Free-Solver-Using-OpenCV/Flow Free Game/5x5.jpg"

img = cv2.imread(path)
n = 5
img = cv2.resize(img,(500, 500))

output = [1,2,2,3,3,1,2,4,3,5,1,2,4,3,5,1,2,4,3,5,1,1,4,5,5]

secw = int(img.shape[1]/n)
sech = int(img.shape[0]/n)

a = b = 0
k = l = 100
m=0
for i in range(0,n):
    for j in range(0,n):
        cv2.rectangle(img, pt1=(a,b),pt2=(k,l),color=col_val[output[m]],thickness=-1)
        print(m)
        a +=100
        k += 100
        m+=1
    a = 0
    k+=100
    b+=100
    l+=100

cv2.imshow("sfdl", img)
cv2.waitKey(0)
cv2.destroyAllWindows()