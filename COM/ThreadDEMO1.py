import time
import pyrealsense2 as rs
import numpy as np
import cv2
import serial
import serial.tools.list_ports
from threading import Thread


def detect_circle_demo(image, img_depth, imgOut):
    kernel = np.ones((11, 11), np.uint8)
    dst = cv2.GaussianBlur(image, (11, 11), 15)
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    Mclose = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Mclose", Mclose)
    circles = cv2.HoughCircles(Mclose, cv2.HOUGH_GRADIENT, 1,
                               100, param1=50, param2=50, minRadius=20, maxRadius=0)
    if not circles is None:
        circles = np.uint16(np.around(circles))
        p = 0
        Dis = []
        for i in circles[0, :]:
            cv2.circle(imgOut, (i[0], i[1]), i[2], (0, 0, 255), 2)
            cv2.circle(imgOut, (i[0], i[1]), 2, (255, 0, 0), 2)  # 圆心
            if i[0] >= 480 or i[1] >= 640:
                continue
            Dis.append(img_depth[i[0], i[1]])
            cv2.putText(imgOut, "Distance:" + str(img_depth[i[0], i[1]]/10) + "cm", (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [255, 0, 255])
        if not len(Dis) == 0:
            p = Dis.index(min(Dis))
        if not circles[0, :][p][0] >= 480 or circles[0, :][p][1] >= 640:
            #print(img_depth[circles[0, :][p][0], circles[0, :][p][1]])
            cv2.putText(imgOut, "TheNearest", (circles[0, :][p][0], circles[0, :][p][1]+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [255, 255, 255])
            return min(Dis)


def initCapture(_W, _H):
    _pc = rs.pointcloud()
    _points = rs.points()

    _pipeline = rs.pipeline()  # 创建一个管道
    config = rs.config()  # Create a config并配置要流​​式传输的管道。
    config.enable_stream(rs.stream.depth, _W, _H, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, _W, _H, rs.format.bgr8, 15)
    '''
    self: pyrealsense2.pyrealsense2.config,
    stream_type: pyrealsense2.pyrealsense2.stream,
    stream_index: int, width: int, height: int,
    format: pyrealsense2.pyrealsense2.format=format.any,
    framerate: int=0
    '''
    # 使用选定的流参数显式启用设备流

    # Start streaming 开启流
    _pipe_profile = _pipeline.start(config)
    '''      流水线循环从设备捕获样本，然后根据每个模块的要求和线程模型，将它们传递到连接的计算机视觉模块和处理模块。
            在循环执行期间，应用程序可以通过调用wait_for_frames（）或poll_for_frames（）来访问摄像机流。
            流循环一直运行到管道停止为止。
    '''
    # Create an align object 创建对其流对象
    # "rs.align" allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    # (对其流)
    align_to = rs.stream.color
    _align = rs.align(align_to)  # 设置为其他类型的流,意思是我们允许深度流与其他流对齐
    # print(type(align))
    _cap = cv2.VideoCapture(0)
    return _pipeline, _align, _pipe_profile, _pc, _points, _cap


"the parameter is the monitoring position"


def led_practice(x_axis, y_axis):
    frames = pipeline.wait_for_frames()  # 等待开启通道,等到新的一组帧集可用为止
    aligned_frames = align.process(frames)  # 将深度框和颜色框对齐
    depth_frame = aligned_frames.get_depth_frame()  # ?获得对齐后的帧数深度数据(图)
    color_frame = aligned_frames.get_color_frame()  # ?获得对齐后的帧数颜色数据(图)
    img_color = np.asanyarray(color_frame.get_data())  # 把图像像素转化为数组
    img_depth = np.asanyarray(depth_frame.get_data())  # 把图像像素转化为数组

    # self: pyrealsense2.pyrealsense2.stream_profile -> rs2::video_stream_profile
    # intrinsics 获取流配置文件的内在属性。
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    # get_extrinsics_to 获取两个配置文件之间的外部转换（代表物理传感器）
    depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(
        color_frame.profile)
    # 获取深度传感器的深度标尺
    depth_sensor = pipe_profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()  # 深度比例系数为： 0.0010000000474974513
#        print("scale:", depth_scale)

    # 由深度到颜色
    depth_pixel = [x_axis, y_axis]  # specified pixel
    # rs2_deproject_pixel_to_point获取实际空间坐标 specified point
    depth_point = rs.rs2_deproject_pixel_to_point(
        depth_intrin, depth_pixel, depth_scale)
#        print(depth_point)
    # perspective conversion
    color_point = rs.rs2_transform_point_to_point(
        depth_to_color_extrin, depth_point)
    # 3D space to XY pixels
    color_pixel = rs.rs2_project_point_to_pixel(color_intrin, color_point)

    pc.map_to(color_frame)  # 将点云映射到给定的颜色帧
    points = pc.calculate(depth_frame)  # 生成深度图的点云和纹理映射
    "points.get_vertices() 检索点云的顶点, asanyarray is similar with array"
    vtx = np.asanyarray(points.get_vertices())  # transfor into XYZ

    return img_color, img_depth


def getMask(_imgBGR):
    _imgHSV = cv2.cvtColor(_imgBGR, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv', _imgHSV)
    _lowHSV_1 = np.array([0, 43, 46])
    _highHSV_1 = np.array([10, 255, 255])
    _mask_0 = cv2.inRange(_imgHSV, _lowHSV_1, _highHSV_1)

    _lowHSV_2 = np.array([156, 43, 46])
    _highHSV_2 = np.array([180, 255, 255])
    _mask_1 = cv2.inRange(_imgHSV, _lowHSV_2, _highHSV_2)

    return _mask_0 + _mask_1


# def writeDis(_minDis, _serial1, _serial2):
#     if _serial2.readline()==('f'+'\n').encode():   
#         if _minDis != None:
#             _serial1.write((str(_minDis)+'\n').encode())
#             print(_serial2.readline())
#         else:
#             _serial1.write((str(0)+'\n').encode())
#             print(_serial2.readline())
def writeDis(_minDis, _serial1, _serial2):
    t1 = time.time()
    if _serial2.readline()==('f'+'\n').encode():   
        if _minDis != None:
            _serial1.write((str('{:>4d}'.format(int(_minDis)))).encode())
            print(_serial2.readline())
        else:
            _serial1.write((str(0)).encode())
            print(_serial2.readline())
    t2 = time.time()
    print(t2-t1)

    

def getKey(_key, _serial):
    if _key&0xFF == ord("f"):
        _serial.write(('f'+'\n').encode())

W = 640
H = 480

plist = list(serial.tools.list_ports.comports())

if len(plist) <= 0:
    print("The Serial port can't find!")
else:
    plist_0 = list(plist[0])
    plist_1 = list(plist[1])
    serialName1 = plist_0[0]
    serialName2 = plist_1[0]
    serialFd1 = serial.Serial(serialName1, 9600, timeout=0.01)
    serialFd2 = serial.Serial(serialName2, 9600, timeout=0.01)

pipeline, align, pipe_profile, pc, points, cap = initCapture(W, H)
while True:
    t3 = time.time()
    imgColor, imgDepth = led_practice(int(W / 2), int(H / 2))
    imgBGR = imgColor.copy()
    mask = getMask(imgBGR)
    imgMask = cv2.bitwise_and(imgBGR, imgBGR, mask=mask)
    minDis = detect_circle_demo(imgMask, imgDepth, imgBGR)

    key = cv2.waitKey(1)

    myThread_1 = Thread(target=writeDis, args=(minDis, serialFd1, serialFd2), name="Write")
    myThread_2 = Thread(target=getKey, args = (key, serialFd1), name="Get")

    myThread_1.start()
    myThread_2.start()

    myThread_2.join()


    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.imshow("Mask", imgMask)

    cv2.namedWindow("BGR", cv2.WINDOW_NORMAL)
    cv2.imshow('BGR', imgBGR)
    t4 = time.time()
    print(t4 - t3)

    if key & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
pipeline.stop()