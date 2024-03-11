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
        board.append(e

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