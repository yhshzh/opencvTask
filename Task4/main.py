import pyrealsense2 as rs
import numpy as np
import cv2
#       "/home/zheng/opencvTask/envs/lib/python3.6/site-packages"
W = 640
H = 480

pc = rs.pointcloud()
points = rs.points()

pipeline = rs.pipeline()  # 创建一个管道
config = rs.config()  # Create a config并配置要流​​式传输的管道。
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 15)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 15)
'''
self: pyrealsense2.pyrealsense2.config, 
stream_type: pyrealsense2.pyrealsense2.stream, 
stream_index: int, width: int, height: int, 
format: pyrealsense2.pyrealsense2.format=format.any, 
framerate: int=0
'''
# 使用选定的流参数显式启用设备流

# Start streaming 开启流
pipe_profile = pipeline.start(config)
'''      流水线循环从设备捕获样本，然后根据每个模块的要求和线程模型，将它们传递到连接的计算机视觉模块和处理模块。
         在循环执行期间，应用程序可以通过调用wait_for_frames（）或poll_for_frames（）来访问摄像机流。
         流循环一直运行到管道停止为止。
'''
# Create an align object 创建对其流对象
# "rs.align" allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
# (对其流)
align_to = rs.stream.color
align = rs.align(align_to)  # 设置为其他类型的流,意思是我们允许深度流与其他流对齐
# print(type(align))
cap = cv2.VideoCapture(0)

"the parameter is the monitoring position"

def detect_circle_demo(image, img_depth, imgOut):
    # dst = cv.bilateralFilter(image, 0, 150, 5)  #高斯双边模糊，不太好调节,霍夫噪声敏感，所以要先消除噪声
    # cv.imshow("1",dst)
    # dst = cv.pyrMeanShiftFiltering(image,5,100)  #均值迁移，EPT边缘保留滤波,霍夫噪声敏感，所以要先消除噪声
    # cv.imshow("2", dst)
    kernel = np.ones((11, 11), np.uint8)
    dst = cv2.GaussianBlur(image,(11,11),15) #使用高斯模糊，修改卷积核ksize也可以检测出来
    # cv.imshow("3", dst)
    gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    Mclose = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel)
    cv2.imshow("Mclose", Mclose)
    circles = cv2.HoughCircles(Mclose,cv2.HOUGH_GRADIENT,1,100,param1=50,param2=50,minRadius=20,maxRadius=0)
    if not circles is None:
        # print(circles[0][0][0])
        circles = np.uint16(np.around(circles))
        # circles = np.where(circles[0][0][0]<480)
        p = 0
        Dis = []
        #print(circles[0,:])
        for i in circles[0,:]:
            cv2.circle(imgOut,(i[0],i[1]),i[2],(0,0,255),2)
            cv2.circle(imgOut,(i[0],i[1]),2,(255,0,0),2)   #圆心
            if i[0]>=480 or i[1]>=640:
                continue
            Dis.append(img_depth[i[0], i[1]]/10)
            cv2.putText(imgOut, "Distance:" + str(img_depth[i[0], i[1]]/10) + "cm", (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [255, 0, 255])
        if not len(Dis) == 0:
            p = Dis.index(min(Dis))
        if not circles[0, :][p][0]>=480 or circles[0, :][p][1]>=640:
            #print(img_depth[circles[0, :][p][0], circles[0, :][p][1]])
            cv2.putText(imgOut, "TheNearest" , (circles[0, :][p][0], circles[0, :][p][1]+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [255, 255, 255])
  

    # cv2.imshow("detect_circle_demo",image)


def led_practice(x_axis, y_axis):
    while True:

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
        depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
        # 获取深度传感器的深度标尺
        depth_sensor = pipe_profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()  # 深度比例系数为： 0.0010000000474974513
#        print("scale:", depth_scale)

        # 由深度到颜色
        depth_pixel = [x_axis, y_axis]  # specified pixel
        # rs2_deproject_pixel_to_point获取实际空间坐标 specified point
        depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_scale)
#        print(depth_point)
        # perspective conversion
        color_point = rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point)
        # 3D space to XY pixels
        color_pixel = rs.rs2_project_point_to_pixel(color_intrin, color_point)

        pc.map_to(color_frame)  # 将点云映射到给定的颜色帧
        points = pc.calculate(depth_frame)  # 生成深度图的点云和纹理映射
        "points.get_vertices() 检索点云的顶点, asanyarray is similar with array"
        vtx = np.asanyarray(points.get_vertices())  # transfor into XYZ
        # tex = np.asanyarray(points.get_texture_coordinates()) # texture map

        # ??????? coordinate transform
        # line by line
        i = W * y_axis + x_axis

        # column by column
        # i = H * x_axis + y_axis

        imgBGR = img_color.copy()
#        print(imgBGR)

        cv2.circle(img_color, (x_axis, y_axis), 8, [255, 0, 255], thickness=-1)

        
        # print(len(img_depth))
        cv2.putText(img_color, "Distance/cm:" + str(img_depth[x_axis, y_axis]/10), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    [255, 0, 255])
        cv2.putText(img_color, "X:" + str(np.float64(vtx[i][0])), (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 255])
        cv2.putText(img_color, "Y:" + str(np.float64(vtx[i][1])), (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 255])
        cv2.putText(img_color, "Z:" + str(np.float64(vtx[i][2])), (80, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 255])
#        print('Distance: ', img_depth[x_axis, y_axis] / 10)
        
        
        imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
        cv2.imshow('hsv', imgHSV)
        lowHSV_1 = np.array([0, 43, 46])
        highHSV_1 = np.array([10, 255, 255])
        mask_0 = cv2.inRange(imgHSV, lowHSV_1, highHSV_1)

        lowHSV_2 = np.array([156, 43, 46])
        highHSV_2 = np.array([180, 255, 255])
        mask_1 = cv2.inRange(imgHSV, lowHSV_2, highHSV_2)

        mask = mask_0 + mask_1

        imgMask = cv2.bitwise_and(imgBGR, imgBGR, mask=mask) # 按位与运算，仅保留HSV [0, 43, 46]~[10, 255, 255]&[156, 43, 46]~[180, 255, 255]
        detect_circle_demo(imgMask, img_depth, imgBGR)

        cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
        cv2.imshow("Mask", imgMask)

        cv2.namedWindow("BGR", cv2.WINDOW_NORMAL)
        cv2.imshow('BGR', imgBGR)


        #cv2.imshow('depth_frame', img_color)
        #cv2.imshow("dasdsadsa", img_depth)

        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break


led_practice(int(W / 2), int(H / 2))
cv2.waitKey(0)
cv2.destroyAllWindows()
pipeline.stop()
