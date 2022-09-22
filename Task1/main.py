import cv2
import numpy as np

def RectangleDetection(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for object in contours:
        area = cv2.contourArea(object)
        perimeter = cv2.arcLength(object, True)
        approx = cv2.approxPolyDP(object, 0.02*perimeter, True)
        CornerNum = len (approx) # 角点数量，用于判断形状

        x, y, w, h = cv2.boundingRect(approx) # x, y, w, h分别为矩形的坐标值和宽度，高度

        if CornerNum == 4:
            cv2.rectangle(imgCopy, (x-5, y-5), (x+w+5, y+h+5), (0, 255, 255), 2) # 标注矩形位置
            cv2.putText(imgCopy, "Rectangle", (x+w//2, y+h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1) 


img = cv2.imread("Task1/P1.jpg")
imgCopy = img.copy()

imgGaus = cv2.GaussianBlur(img, (7, 7), 1.5)

lowBGR = np.array([0,0,50])
highBGR = np.array([255,255,255])
mask = cv2.inRange(imgGaus,lowBGR,highBGR)
imgMask = cv2.bitwise_and(imgGaus,imgGaus,mask=mask) # 按位与运算，仅保留BGR在[0, 0, 50]及以上的像素点


imgGray = cv2.cvtColor(imgMask, cv2.COLOR_BGR2GRAY)
imgGaus2 = cv2.GaussianBlur(imgGray, (3, 3), sigmaX = 0, sigmaY = 0)
imgCanny = cv2.Canny(imgGaus2, 60, 60)

RectangleDetection(imgCanny)

cv2.imshow("Origin", img)
cv2.imshow('Mask',imgMask)
cv2.imshow("Gray", imgGray)
cv2.imshow("Gaus", imgGaus2)
cv2.imshow("Canny", imgCanny)
cv2.imshow("Rectangle Detection", imgCopy)

cv2.waitKey(0)