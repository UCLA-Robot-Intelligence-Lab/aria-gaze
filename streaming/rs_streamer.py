import pyrealsense2 as rs
import numpy as np
import cv2
from cv2 import aruco
import imageio

class MarkSearch:

    def __init__(self):
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters =  aruco.DetectorParameters()

        self.detector = aruco.ArucoDetector(self.dictionary, self.parameters)

    def find_marker(self, frame):
        """
        Obtain marker id list from still image
        """
        # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # corners, ids, rejectedImgPoints = self.detector.detectMarkers(gray)

        # if ids is None:
        #     return (None,None),None

        # ids = ids.flatten()

        # # loop over the detected ArUCo corners
        # for (markerCorner, markerID) in zip(corners, ids):
        #     # extract the marker corners (which are always returned in
        #     # top-left, top-right, bottom-right, and bottom-left order)
        #     corners = markerCorner.reshape((4, 2))
        #     (topLeft, topRight, bottomRight, bottomLeft) = corners
        #     # convert each of the (x, y)-coordinate pairs to integers

        #     #u = np.mean((topLeft[0], bottomRight[0])).astype(int)
        #     #v = np.mean((topLeft[1], bottomRight[1])).astype(int)
        #     u = np.mean((topLeft[0], bottomRight[0]))
        #     v = np.mean((topLeft[1], bottomRight[1]))

        #     cv2.circle(image, (int(u),int(v)), 5, (0,0,255), -1)
        #     cv2.imshow("rgb", image)
        #     cv2.waitKey(1)

        #     return (u,v), image

        # 1- convert frame from BGR to HSV
        HSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        # 2- define the range of red
        lower=np.array([-10, 100, 100])
        upper=np.array([10, 255, 255])

        #check if the HSV of the frame is lower or upper red
        Red_mask = cv2.inRange(HSV,lower, upper)
        result = cv2.bitwise_and(frame, frame, mask = Red_mask)

        # Draw rectangular bounded line on the detected red area
        (contours, _) = cv2.findContours(Red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for _,contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300): #to remove the noise
                # Constructing the size of boxes to be drawn around the detected red area
                x,y,w,h = cv2.boundingRect(contour)
                #frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                frame = cv2.circle(frame, (int(x+ w/2),int(y+ h/2)), radius=5, color=(255, 0, 0), thickness=-1)

                u = int(x + w/2)
                v = int(y + h/2)
                return (u,v), frame
        return (None,None), None


class RealsenseStreamer():
    def __init__(self, serial_no=None):

        # in-hand : 317222072157
        # external: 317422075456

        # Configure depth and color streams
        self.pipeline = rs.pipeline()


        self.config = rs.config()

        if serial_no is not None:
            self.config.enable_device(serial_no)

        self.width = 640
        self.height = 480

        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)

        self.align_to_color = rs.align(rs.stream.color)

        # Start streaming
        self.pipe_profile = self.pipeline.start(self.config)

        profile = self.pipeline.get_active_profile()

        ## Configure depth sensor settings
        depth_sensor = profile.get_device().query_sensors()[0]
        depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
        depth_sensor.set_option(rs.option.depth_units, 0.001)
        preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
        for i in range(int(preset_range.max)):
            visualpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
            if visualpreset == "Default":
                depth_sensor.set_option(rs.option.visual_preset, i)

        color_sensor = profile.get_device().query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure, 1)

        self.serial_no = serial_no

        if self.serial_no == '317422075456':
            color_sensor.set_option(rs.option.exposure, 140)

        # Intrinsics & Extrinsics
        frames = self.pipeline.wait_for_frames()
        frames = self.align_to_color.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        self.depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        self.color_intrin = color_frame.profile.as_video_stream_profile().intrinsics

        self.depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
        self.colorizer = rs.colorizer()

        self.K = np.array([[self.depth_intrin.fx, 0, self.depth_intrin.ppx],
                           [0, self.depth_intrin.fy, self.depth_intrin.ppy],
                           [0, 0, 1]])

        self.dec_filter = rs.decimation_filter()
        self.spat_filter = rs.spatial_filter()
        self.temp_filter = rs.temporal_filter()
        self.hole_filling_filter = rs.hole_filling_filter()
        print("camera started")

    def deproject(self, px, depth_frame):
        u,v = px
        depth = depth_frame.get_distance(u,v)
        xyz = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [u,v], depth)
        return xyz

    def capture_rgb(self):
        color_frame = None
        while True:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame is not None:
                color_image = np.asanyarray(color_frame.get_data())
                break
        return color_image

    def filter_depth(self, depth_frame):
        #filtered = self.dec_filter.process(depth_frame)
        #filtered = self.spat_filter.process(filtered)

        filtered = depth_frame
        #filtered = self.hole_filling_filter.process(filtered)
        #filtered = self.temp_filter.process(filtered)
        #filtered = self.spat_filter.process(filtered)
        return filtered.as_depth_frame()

    def capture_rgbd(self):
        frame_error = True
        while frame_error:
            try:
                frames = self.align_to_color.process(frames)  
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                frame_error = False
            except:
                frames = self.pipeline.wait_for_frames()
                continue
        color_image = np.asanyarray(color_frame.get_data())
        depth_frame = self.filter_depth(depth_frame)
        depth_image = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
        return color_frame, color_image, depth_frame, depth_image

    def stop_stream(self):
        self.pipeline.stop()

    def show_image(self, image):
        cv2.imshow('img', image)
        cv2.waitKey(0)

if __name__ == '__main__':
    #realsense_streamer  = RealsenseStreamer('317222072157')
    realsense_streamer = RealsenseStreamer('317422075456') #317422074281 small
    marker_search = MarkSearch()

    frames = []
    while True:
        _, rgb_image, depth_frame, depth_img = realsense_streamer.capture_rgbd()
        cv2.waitKey(1)
        cv2.imshow('img', rgb_image)
        (u,v), vis = marker_search.find_marker(rgb_image)
        print(u,v)
        cv2.imshow('img', np.hstack((depth_img, vis)))

