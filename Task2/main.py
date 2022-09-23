import cv2
import numpy as np

cap = cv2.VideoCapture(0)

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

    cv2.imshow("Mask", imgMask)

    if cv2.waitKey(1)&0xFF == ord("q"):
        cap.release()
        break


    