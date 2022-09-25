import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def detect_circle_demo(image):
    # dst = cv.bilateralFilter(image, 0, 150, 5)  #高斯双边模糊，不太好调节,霍夫噪声敏感，所以要先消除噪声
    # cv.imshow("1",dst)
    # dst = cv.pyrMeanShiftFiltering(image,5,100)  #均值迁移，EPT边缘保留滤波,霍夫噪声敏感，所以要先消除噪声
    # cv.imshow("2", dst)
    dst = cv2.GaussianBlur(image,(13,15),15) #使用高斯模糊，修改卷积核ksize也可以检测出来
    # cv.imshow("3", dst)
    gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    circles = np.ndarray(1)
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
    print(circles)
    if not circles is None :
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(image,(i[0],i[1]),i[2],(0,0,255),2)
            cv2.circle(image,(i[0],i[1]),2,(255,0,0),2)   #圆心

        cv2.imshow("detect_circle_demo",image)


while cap.isOpened():
    ret, imgBGR = cap.read()
    # cv2.imshow("Origin", imgBGR)
    imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
    # cv2.imshow("HSV", imgHSV)

    lowHSV_1 = np.array([0, 43, 46])
    highHSV_1 = np.array([10, 255, 255])
    mask_0 = cv2.inRange(imgHSV, lowHSV_1, highHSV_1)

    lowHSV_2 = np.array([156, 43, 46])
    highHSV_2 = np.array([180, 255, 255])
    mask_1 = cv2.inRange(imgHSV, lowHSV_2, highHSV_2)

    mask = mask_0 + mask_1

    imgMask = cv2.bitwise_and(imgBGR, imgBGR, mask=mask) # 按位与运算，仅保留HSV [0, 43, 46]~[10, 255, 255]&[156, 43, 46]~[180, 255, 255]

    cv2.putText(imgMask, "HSV:[0, 43, 46]~[10, 255, 255]&[156, 43, 46]~[180, 255, 255]", (0, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    detect_circle_demo(imgBGR)
    
    cv2.imshow("Mask", imgMask)

    if cv2.waitKey(1)&0xFF == ord("q"):
        cap.release()
        break


    