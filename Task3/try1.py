import cv2
import numpy as np



def RectangleDetection(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for object in contours:
        area = cv2.contourArea(object)
        perimeter = cv2.arcLength(object, True)
        approx = cv2.approxPolyDP(object, 0.02*perimeter, True) # 最优拟合多边形，用于计算角点数量获取矩形
        CornerNum = len (approx) # 角点数量，用于判断形状

        x, y, w, h = cv2.boundingRect(approx) # x, y, w, h分别为矩形的坐标值和宽度，高度

        if area >= 50000:
            return approx



source0 = cv2.VideoCapture("Task3/source0.MOV")
source2 = cv2.VideoCapture("Task3/source2.mp4")

lowHSV = np.array([0, 0, 0])
upHSV = np.array([180, 255, 46])
kernel = np.ones((11, 11), np.uint8)
i = 0

frameCounter = 0

while source0.isOpened() and source2.isOpened():
    ret1, imgBGR = source0.read()

    frameCounter += 1
    if frameCounter == int(source0.get(cv2.CAP_PROP_FRAME_COUNT)):
        frameCounter = 0
        source0.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret2, imgS2 = source2.read()
    if not ret1 or not ret2:
        source0.release()
        source2.release()
        break

    imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(imgHSV, lowHSV, upHSV)

    res = cv2.bitwise_and(imgBGR, imgBGR, mask = mask)
    ret, imgThreShold = cv2.threshold(res, 1, 255, cv2.THRESH_BINARY)
    imgGray = cv2.cvtColor(imgThreShold, cv2.COLOR_BGR2GRAY)
    Mclose = cv2.morphologyEx(imgGray,cv2.MORPH_CLOSE,kernel)

    imgGray[1200:,:]=0

    i = i + 1
    pointsSource0 = np.float32(RectangleDetection(imgGray).reshape(4,2))
    pointsSource0R = np.float32(RectangleDetection(imgGray).reshape(4,2))

    # 找出左上角的点，并对坐标重新排序
    for i in range(4):
        for j in range (4):
            if (pointsSource0[i][0]+pointsSource0[i][1] < pointsSource0[j][0]+pointsSource0[j][1]):
                i = j
    for k in range(4):
        pointsSource0R[k] = pointsSource0[(k+i-2)%4]


    pointsSource2 = np.array([[0, 0], [0, imgS2.shape[0]], [imgS2.shape[1], imgS2.shape[0]], [imgS2.shape[1], 0]], dtype = np.float32)

    # 以下为仿射变换部分
    retval = cv2.getPerspectiveTransform(pointsSource2, pointsSource0R)
    imgWarp = cv2.warpPerspective(imgS2, retval, (1080, 1920))

    # 两图像叠加
    Added = cv2.add(imgBGR, imgWarp)

    cv2.namedWindow("Add", cv2.WINDOW_NORMAL)
    cv2.imshow("Add", Added)

    if cv2.waitKey(6) & 0xFF == ord("q"):
        source0.release()
        source2.release()
        break
cv2.destroyAllWindows()