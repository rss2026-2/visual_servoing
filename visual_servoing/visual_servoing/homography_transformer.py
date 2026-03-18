#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from std_msgs.msg import String
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from vs_msgs.msg import ConeLocation, ConeLocationPixel
from geometry_msgs.msg import Point

# The following collection of pixel locations and corresponding relative
# ground plane locations are used to compute our homography matrix

# PTS_IMAGE_PLANE units are in pixels
# see README.md for coordinate frame description

######################################################
# DUMMY POINTS -- ENTER YOUR MEASUREMENTS HERE
PTS_IMAGE_PLANE = [[497, 268], # avg over 5 clicks (492 267, 492 267, 502 270, 501 267, 502 269 )
                   [395, 148], # avg over 5 clicks (397 148, 395 149, 394 148, 394 148, 395 147)
                   [160.0, 175], # avg over 5 clicks  (162 175, 158 175, 160 174, 158 175, 160 174)
                   [56, 159]]  # avg over 5 clicks (56 158, 56 159, 56 159, 57 158, 54 160)
######################################################

# PTS_GROUND_PLANE units are in inches
# car looks along positive x axis with positive y axis to left

######################################################
# DUMMY POINTS -- ENTER YOUR MEASUREMENTS HERE
PTS_GROUND_PLANE = [[17.99, -10.51], # avg over 3 measurements  18 in x 10.5 in
                    [134.5,  -37.0], # avg over 3 measurements
                    [57.50, 24.25], # avg over 3 measurements
                    [99.0, 77.0]]  # avg over 3 measurements
######################################################

METERS_PER_INCH = 0.0254


class HomographyTransformer(Node):
    def __init__(self):
        super().__init__("homography_transformer")

        self.cone_pub = self.create_publisher(ConeLocation, "/relative_cone", 10)
        self.marker_pub = self.create_publisher(Marker, "/cone_marker", 1)
        self.cone_px_sub = self.create_subscription(ConeLocationPixel, "/relative_cone_px", self.cone_detection_callback, 1)
        # added
        self.click_px_sub = self.create_subscription(Point, "/cone_debug_img_mouse_left", self.click_callback, 1)

        if not len(PTS_GROUND_PLANE) == len(PTS_IMAGE_PLANE):
            rclpy.logerr("ERROR: PTS_GROUND_PLANE and PTS_IMAGE_PLANE should be of same length")

        # Initialize data into a homography matrix

        np_pts_ground = np.array(PTS_GROUND_PLANE)
        np_pts_ground = np_pts_ground * METERS_PER_INCH
        np_pts_ground = np.float32(np_pts_ground[:, np.newaxis, :])

        np_pts_image = np.array(PTS_IMAGE_PLANE)
        np_pts_image = np_pts_image * 1.0
        np_pts_image = np.float32(np_pts_image[:, np.newaxis, :])

        self.h, err = cv2.findHomography(np_pts_image, np_pts_ground)

        self.get_logger().info("Homography Transformer Initialized")

    def cone_detection_callback(self, msg):
        # Extract information from message
        u = msg.u
        v = msg.v

        # Call to main function
        x, y = self.transformUvToXy(u, v)
        y += 0.06

        # Publish relative xy position of object in real world
        relative_xy_msg = ConeLocation()
        relative_xy_msg.x_pos = x
        relative_xy_msg.y_pos = y

        self.draw_marker(x, y, '/base_link')
        self.cone_pub.publish(relative_xy_msg)

    def click_callback(self, msg):
        u = msg.x
        v = msg.y
        

        x, y = self.transformUvToXy(u, v)
        
        self.get_logger().info(f'{u=}, {v=}')
        self.get_logger().info(f'{x=}, {y=}')
        
        # Publish relative xy position of object in real world
        relative_xy_msg = ConeLocation()
        relative_xy_msg.x_pos = x
        relative_xy_msg.y_pos = y
        
        self.draw_marker(x, y, "/base_link")
        self.cone_pub.publish(relative_xy_msg)
        

    def transformUvToXy(self, u, v):
        """
        u and v are pixel coordinates.
        The top left pixel is the origin, u axis increases to right, and v axis
        increases down.

        Returns a normal non-np 1x2 matrix of xy displacement vector from the
        camera to the point on the ground plane.
        Camera points along positive x axis and y axis increases to the left of
        the camera.

        Units are in meters.
        """
        homogeneous_point = np.array([[u], [v], [1]])
        xy = np.dot(self.h, homogeneous_point)
        scaling_factor = 1.0 / xy[2, 0]
        homogeneous_xy = xy * scaling_factor
        x = homogeneous_xy[0, 0]
        y = homogeneous_xy[1, 0]
        # self.get_logger().info(f"x-value: {x} \n y-value: {y}")
        return x, y

    def draw_marker(self, cone_x, cone_y, message_frame):
        """
        Publish a marker to represent the cone in rviz.
        (Call this function if you want)
        """
        marker = Marker()
        marker.header.frame_id = message_frame
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = .2
        marker.scale.y = .2
        marker.scale.z = .2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = .5
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = cone_x
        marker.pose.position.y = cone_y
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    homography_transformer = HomographyTransformer()
    rclpy.spin(homography_transformer)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
