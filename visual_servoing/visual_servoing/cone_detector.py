#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point #geometry_msgs not in CMake file
from std_msgs.msg import Bool
from vs_msgs.msg import ConeLocationPixel

# import your color segmentation algorithm; call this function in ros_image_callback!
from computer_vision.color_segmentation import cd_color_segmentation


class ConeDetector(Node):
    """
    A class for applying your cone detection algorithms to the real robot.
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: /relative_cone_px (ConeLocationPixel) : the coordinates of the cone in the image frame (units are pixels).
    """

    def __init__(self):
        super().__init__("cone_detector")
        # set line follower image crop parameters
        self.declare_parameter("y_min", 0.54)
        self.declare_parameter("y_max", 0.638)
        # set proximity check parameters
        self.declare_parameter("prox_threshold", 0.0)
        
        self.y_min = self.get_parameter("y_min").get_parameter_value().double_value
        self.y_max = self.get_parameter("y_max").get_parameter_value().double_value
        self.prox_threshold = self.get_parameter("prox_threshold").get_parameter_value().double_value

        # Subscribe to ZED camera RGB frames
        self.cone_pub = self.create_publisher(ConeLocationPixel, "/relative_cone_px", 10)
        self.debug_pub = self.create_publisher(Image, "/cone_debug_img", 10)
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
        self.proximity_pub = self.create_publisher(Bool, "/proximity_check", 10)
        self.bridge = CvBridge()  # Converts between ROS images and OpenCV Images

        self.get_logger().info("Cone Detector Initialized")

    def image_callback(self, image_msg):
        # Apply your imported color segmentation function (cd_color_segmentation) to the image msg here
        # From your bounding box, take the center pixel on the bottom
        # (We know this pixel corresponds to a point on the ground plane)
        # publish this pixel (u, v) to the /relative_cone_px topic; the homography transformer will
        # convert it to the car frame.

        #################################
        # YOUR CODE HERE
        # detect the cone and publish its
        # pixel location in the image.
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        #################################

        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        cone_template = cv2.imread("/root/racecar_ws/src/visual_servoing/visual_servoing/visual_servoing/computer_vision/test_images_cone/cone_template.png")



        # crop image
        image_width, image_height = image.shape[1], image.shape[0]
        image_y_min, image_y_max = int(self.y_min * image_height), int(self.y_max * image_height)
        
        cropped_image = image[image_y_min:image_y_max,:]
        cropped_image_height = cropped_image.shape[0]
        
        bbox = cd_color_segmentation(cropped_image, cone_template)


        if bbox is not None:
            cv2.line(image, pt1 = (0, int(self.y_min * image_height)), pt2 = (image_width-1, int(self.y_min * image_height)), color = (0, 0, 255), thickness = 2)
            cv2.line(image, pt1 = (0, int(self.y_max * image_height)), pt2 = (image_width-1, int(self.y_max * image_height)), color = (0, 0, 255), thickness = 2)

            bbox_uncropped = [(bbox[0][0], bbox[0][1] + int(self.y_min * image_height)), (bbox[1][0], bbox[1][1] +  int(self.y_min * image_height))]
            cv2.rectangle(image, bbox_uncropped[0], bbox_uncropped[1], (255,0,0), 2)

            debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self.debug_pub.publish(debug_msg)
            
            bottom_center_px = get_bottom_center_of_bounds(bbox_uncropped)

            cone_msg = ConeLocationPixel()
            cone_msg.u, cone_msg.v = bottom_center_px

            proximity_msg = Bool()
            if cone_msg.v > cropped_image_height * (1 - self.prox_threshold):
                proximity_msg.data = True
            else:
                proximity_msg.data = False
            self.proximity_pub.publish(proximity_msg)
            self.cone_pub.publish(cone_msg)

def get_bottom_center_of_bounds(bounding_box):
    top_left_px, bottom_right_px = bounding_box
    top_left_x, _ = top_left_px
    bottom_right_x, bottom_right_y = bottom_right_px

    return ( float((top_left_x + bottom_right_x) / 2), float(bottom_right_y) )

def main(args=None):
    rclpy.init(args=args)
    cone_detector = ConeDetector()
    rclpy.spin(cone_detector)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
