import numpy as np
import cv2
import math


# 计算两点之间线段的距离

def __line_magnitude(x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
    return lineMagnitude


def __point_to_line_distance(point, line):
    px, py = point
    x1, y1, x2, y2 = line
    line_magnitude = __line_magnitude(x1, y1, x2, y2)
    if line_magnitude < 0.00000001:
        return 9999
    else:
        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (line_magnitude * line_magnitude)
        if (u < 0.00001) or (u > 1):
            # 点到直线的投影不在线段内, 计算点到两个端点距离的最小值即为"点到线段最小距离"
            ix = __line_magnitude(px, py, x1, y1)
            iy = __line_magnitude(px, py, x2, y2)
            if ix > iy:
                distance = iy
            else:
                distance = ix
        else:
            # 投影点在线段内部, 计算方式同点到直线距离, u 为投影点距离x1在x1x2上的比例, 以此计算出投影点的坐标
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            distance = __line_magnitude(px, py, ix, iy)
        return distance


point_1_X = 0
point_1_Y = 0

point_2_X = np.random.randint(100, 200)
point_2_Y = np.random.randint(300, 400)

point_3_X = np.random.randint(200, 300)
point_3_Y = np.random.randint(100, 200)

point_4_X = np.random.randint(500, 600)
point_4_Y = np.random.randint(200, 300)

point_1 = (point_1_X, point_1_Y)
point_2 = (point_2_X, point_2_Y)
point_3 = (point_3_X, point_3_Y)
point_4 = (point_4_X, point_4_Y)
point_end = (600, point_3_Y + (point_4_Y-point_3_Y) /
             (point_4_X-point_3_X)*(600 - point_3_X))

img = np.zeros((400, 600, 3), dtype=np.uint8)
img[:][:] = 255

cv2.line(img, point_1, point_2, (0, 0, 0))
cv2.line(img, point_2, point_3, (0, 0, 0))
cv2.line(img, point_3, point_4, (0, 0, 0))
#cv2.line(img, point_1, point_2, (0, 0, 0))


imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Blue : imgHSV[:][:] = [124, 255, 255]
# Red  : imgHSV[:][:] = [180, 255, 255]

# point1 and point 2 -> line1
line1 = point_1+point_2
# point2 and point 3 -> line2
line2 = point_2+point_3
# point3 and point 4 -> line3
line3 = point_3+point_4

lineEnd = point_3+point_end

# line1
for x in range(point_1_X, point_2_X):
    for y in range(400):
        point = [x, y]
        dis = min(__point_to_line_distance(point, line1), __point_to_line_distance(
            point, line2), __point_to_line_distance(point, line3))
        line_y = point_1_Y + (point_2_Y-point_1_Y) / \
            (point_2_X-point_1_X)*(x - point_1_X)
        if y >= line_y:
            imgHSV[y][x] = [124, round(dis/500*255), 255]
        if y < line_y:
            imgHSV[y][x] = [180, round(dis/500*255), 255]

# line2
for x in range(point_2_X, point_3_X):
    for y in range(400):
        point = [x, y]
        dis = min(__point_to_line_distance(point, line1), __point_to_line_distance(
            point, line2), __point_to_line_distance(point, line3))
        line_y = point_2_Y + (point_3_Y-point_2_Y) / \
            (point_3_X-point_2_X)*(x - point_2_X)
        if y >= line_y:
            imgHSV[y][x] = [124, round(dis/500*255), 255]
        if y < line_y:
            imgHSV[y][x] = [180, round(dis/500*255), 255]

# line3
for x in range(point_3_X, 600):
    for y in range(400):
        point = [x, y]
        dis = min(__point_to_line_distance(point, line1), __point_to_line_distance(
            point, line2), __point_to_line_distance(point, lineEnd))
        line_y = point_3_Y + (point_4_Y-point_3_Y) / \
            (point_4_X-point_3_X)*(x - point_3_X)
        if y >= line_y:
            imgHSV[y][x] = [124, round(dis/500*255), 255]
        if y < line_y:
            imgHSV[y][x] = [180, round(dis/500*255), 255]


imgBGR = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2BGR)
cv2.line(imgBGR, point_1, point_2, (0, 0, 0))
cv2.line(imgBGR, point_2, point_3, (0, 0, 0))
cv2.line(imgBGR, point_3, point_4, (0, 0, 0))

# img[point_1_X, point_1_Y] = 0
# img[point_2_X, point_2_Y] = 0
# img[point_3_X, point_3_Y] = 0
# img[point_4_X, point_4_Y] = 0

cv2.namedWindow("source", cv2.WINDOW_NORMAL)
cv2.imshow("source", imgBGR)
cv2.waitKey(0)
