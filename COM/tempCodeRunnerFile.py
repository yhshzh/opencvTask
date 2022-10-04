      pipeline, align, pipe_profile, pc, points, cap = initCapture(W, H)
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
