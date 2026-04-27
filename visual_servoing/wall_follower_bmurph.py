#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from rcl_interfaces.msg import SetParametersResult

from viz_utils.visualization_tools import VisualizationTools

# added
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from ackermann_msgs.msg import AckermannDrive
import random


class WallFollower(Node):

    def __init__(self):
        super().__init__("wall_follower")
        # Declare parameters to make them available for use
        # DO NOT MODIFY THIS!
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("side", 1)
        self.declare_parameter("velocity", 1.0)
        self.declare_parameter("desired_distance", 1.0)
        # added
        self.declare_parameter("wall_topic", "/wall")
        self.declare_parameter("car_length", 0.325)
        self.declare_parameter("max_steering_angle", 0.34)

        # Fetch constants from the ROS parameter server
        # DO NOT MODIFY THIS! This is necessary for the tests to be able to test varying parameters!
        self.SCAN_TOPIC = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.DRIVE_TOPIC = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.SIDE = self.get_parameter('side').get_parameter_value().integer_value
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value
        # added
        self.WALL_TOPIC = self.get_parameter('wall_topic').get_parameter_value().string_value
        self.CAR_LENGTH = self.get_parameter('car_length').get_parameter_value().double_value
        self.MAX_STEERING_ANGLE = self.get_parameter('max_steering_angle').get_parameter_value().double_value

        self.CLIP = 2.5
        self.INTENSITY_THRESH = 0.9
        self.LOOKAHEAD = 0.8
        self.TURNING_SPEED = self.VELOCITY * 0.95


        self.past_error = 0
        self.past_time = 0
        # self.kp = 0.5 # proportional feedback gain
        self.kd = 0.08 # derivative feedback gain

        # This activates the parameters_callback function so that the tests are able
        # to change the parameters during testing.
        # DO NOT MODIFY THIS!
        self.add_on_set_parameters_callback(self.parameters_callback)

        # TODO: Initialize your publishers and subscribers here

        # Subscribe to the /scan topic
        self.scan_subscriber_ = self.create_subscription(LaserScan, self.SCAN_TOPIC, self.listener_callback, 10)

        # Publish to the /wall topic
        self.wall_publisher_ = self.create_publisher(Marker, self.WALL_TOPIC, 1)

        # Publish to the /des topic
        self.desired_publisher_ = self.create_publisher(Marker, '/des', 1)

        # TODO: Write your callback functions here
        self.drive_publisher_ = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 1)

    def listener_callback(self, scan):
        """
        Listener callback function for the scan subscriber.

        Args:
            scan - The LaserScan coming in from the subscriber.
        """
        coords = self.get_valid_coords(scan)
        wall_start, wall_end, desired_start, desired_end = self.compute_wall(coords, epsilon=0.2)

        if wall_start is None:
            return

        # Line goes from start_x to end_x and start_y to end_y
        x = np.linspace(wall_start[0], wall_end[0], num=20)
        y = np.linspace(wall_start[1], wall_end[1], num=20)

        # Plot line
        VisualizationTools.plot_line(x, y, self.wall_publisher_)
        # self.get_logger().info(f'Line plotted from ({min_x}, {min_y}) to ({max_x}, {max_y}) on side {self.SIDE}')

        x = np.linspace(desired_start[0], desired_end[0], num=20)
        y = np.linspace(desired_start[1], desired_end[1], num=20)
        VisualizationTools.plot_line(x, y, self.desired_publisher_)

        drive_msg = self.update_control((wall_start, wall_end), (desired_start, desired_end))
        self.drive_publisher_.publish(drive_msg)

    def update_control(self, wall_coords, desired_coords):
        """
        Function to compute updated control values to publish.

        Args:
            wall_coords: The start and end coordinates of the wall segment we're following.
        """
        drive_msg = AckermannDriveStamped()

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'base_link'

        drive = AckermannDrive()
        drive.steering_angle = self.compute_feedback_angle(wall_coords, desired_coords)

        path_vector_norm = (desired_coords[1] - desired_coords[0]) / np.linalg.norm(desired_coords[1] - desired_coords[0])

        # angle_to_horizontal = np.arccos( (1, 0) dot path_vector_norm = np.arccos(path_vector_norm.x) )
        angle_to_horizontal = np.arccos(path_vector_norm[0])
        if abs(angle_to_horizontal) < 0.50:
            drive.speed = self.VELOCITY * 2
        else:
            drive.speed = self.TURNING_SPEED

        drive_msg.header = header
        drive_msg.drive = drive

        return drive_msg

    def compute_feedback_angle(self, wall_coords, desired_coords):
        """
        Function to compute the angle using feedback control.

        Args:
            wall_coords: The start and end coordinates of the wall segment we're following.
            wall_coords: The start and end coordinates of the desired path segment we're following.
        """
        # Wall is always pointing in positive x and direction of turn (stright if vertical orientation)
        wall_start, wall_end = wall_coords
        desired_start, desired_end = desired_coords

        # Get a target point along path line
        unit_vector = (desired_end - desired_start) / np.linalg.norm(desired_end - desired_start)
        target_point = desired_start + (unit_vector * self.LOOKAHEAD)
        # Get the distance to that point
        lookahead_dist = np.linalg.norm(target_point)


        # compute the steering angle
        # delta = arctan(2*L*sin(eta)/lookahead_dist)
        # where sin(eta) = y_target / lookahead_distance^2
        delta = np.arctan2(2 * self.CAR_LENGTH * target_point[1], (lookahead_dist**2))

        # compute the error
        current_time = self.get_clock().now().nanoseconds * 1e-9
        dt = current_time - self.past_time
        # error = distance from desired = desired_start.y
        error = desired_start[1]
        error_deriv = (error - self.past_error) / dt
        delta_kd = self.kd * error_deriv

        # Combine error and pure pursuit calculation
        delta = delta + delta_kd
        # Clip to the max steering distance
        delta = np.clip(delta, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        self.past_error = error
        self.past_time = current_time

        return delta
    


    def compute_wall(self, coords, epsilon=0.5):
        """
        Function to compute the wall.

        Args:
            coords: Sliced coordinate data from scanned LIDAR data. Shape: (2, num_points)
            epsilon: Acceptable margin for finding inliers.
        """
        # Use the RANSAC algorithm to find points representing line of best fit
        best_line = self.compute_line_RANSAC(coords, epsilon)

        # If best_line is none, didn't find enough points to compute best fit
        if best_line == None:
            self.get_logger().info("Can't find a wall with less than 2 points!")
            return None, None, None, None

        point1, point2 = best_line

        # Create normalized direction vector for the wall
        wall_vector = point2 - point1
        wall_vector_norm = wall_vector / np.linalg.norm(wall_vector)

        # Determine whether wall has vertical orientation w.r.t. forward direction of car
        has_vertical_orientation = abs(wall_vector_norm[0]) >= abs(wall_vector_norm[1])

        # Ensure that wall vector is always pointing in positive x (if vertical) or horizontal direction of turn
        if has_vertical_orientation and wall_vector_norm[0] < 0:
            wall_vector_norm = -wall_vector_norm
        elif not has_vertical_orientation:
            # Flip wall orientation if car is following left wall and wall is pointing left
            if wall_vector_norm[1] > 0 and self.SIDE == 1:
                wall_vector_norm = -wall_vector_norm
            # Flip wall orientation if car is following right wall and wall is pointing right
            elif wall_vector_norm[1] < 0 and self.SIDE == -1:
                wall_vector_norm = -wall_vector_norm

        # Logic for wall with vertical orientation
        if has_vertical_orientation:
            # x = o_x + t * wall_norm_x --> t = (x - o_x) / wall_norm_x
            # Find the start of the wall, which should be at x = 0
            t_start = -point1[0] / wall_vector_norm[0]
            # Find the end of the wall, which is the furthest point seen by LIDAR (clipped by self.CLIP)
            max_x = min(np.max(coords[0, :]), self.CLIP)
            t_end = (max_x - point1[0]) / wall_vector_norm[0]

        # Logic for horizontal wall orientation
        else:
            # y = o_y + t * wall_norm_y --> t = (y - o_y) / wall_norm_y
            # Find the start of the wall, which should be at y = 0
            t_start = -point1[1] / wall_vector_norm[1]

            # Find end of the wall based on what side the wall will be followed by
            if self.SIDE == 1:
                # Wall will be followed by left side of the car
                # largest y will be negative
                bound_y = max(np.min(coords[1, :]), -self.CLIP)
            else:
                # Wall will be followed by right side of the car
                # largest y will be positive
                bound_y = min(np.max(coords[1, :]), self.CLIP)

            # Find end of the wall
            t_end = (bound_y - point1[1]) / wall_vector_norm[1]

        # Convert back to cartesian map coordinates
        wall_start = point1 + t_start * wall_vector_norm
        wall_end = point1 + t_end * wall_vector_norm

        # Find desired path to follow
        if has_vertical_orientation:
            desired_start = wall_start - np.array([0, self.SIDE * self.DESIRED_DISTANCE])
            desired_end = wall_end - np.array([0, self.SIDE * self.DESIRED_DISTANCE])
        else:
            desired_start = wall_start - np.array([self.DESIRED_DISTANCE, 0])
            desired_end = wall_end - np.array([self.DESIRED_DISTANCE, 0])

        return wall_start, wall_end, desired_start, desired_end

    def compute_line_RANSAC(self, coords, epsilon, num_iterations=500):
        """
        Function to compute the infinite wall line using the RANSAC algorithm given coordinate data.

        Args:
            coords: Sliced coordinate data from scanned LIDAR data. Shape: (2, num_points)
            epsilon: Acceptable margin for finding inliers.
        """
        points = coords.T # Shape: (num_points, 2)
        num_points = points.shape[0]

        # Check if there are less than 2 points
        if num_points < 2:
            # Can't form a line, return None
            return None

        # Initialize max number of inliers
        max_inliers = -1
        # Initialize best line
        best_line = ((0,0), (0,0))

        for _ in range(num_iterations):
            # Randomly sample 2 points from the set of points
            idx1, idx2 = random.sample(range(num_points), 2)
            point1, point2 = points[idx1, :], points[idx2, :]

            # Create a vector between these two points
            line_vector = point2 - point1

            # If two points are the same, skip
            if np.linalg.norm(line_vector) == 0:
                continue

            # Find vector perpendicular to line (and normalize)
            normal_vector = np.array([line_vector[1], -line_vector[0]]) / np.linalg.norm(line_vector)

            # For all points, find distance between vector and point on line
            distance_vectors = points - point2
            distances = np.abs(np.dot(distance_vectors, normal_vector)) # (num_points, 2) dot (1, 2)

            # Count the number of inliers by counting the number of distances less than epsilon
            num_inliers = (distances < epsilon).sum()

            # Check if number of inliers is greater than max_inlier
            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_line = (point1, point2)

        # Return two points making up the best line
        return best_line


    def get_valid_coords(self, scan):
        """
        Helper function to compute valid coordinates to analyze given LaserScan data.

        Args:
            scan: The LaserScan data we are processing.
        """
        # Convert ranges into numpy array
        ranges = np.array(scan.ranges, dtype="float32")
        # Convert intensities into numpy array
        intensities = np.array(scan.intensities, dtype="float32")
        # Normalize intensities by the max value scanned
        intensities_norm = intensities / np.median(intensities)
        # self.get_logger().info(f'{intensities_norm=}')

        # Compute angles array using linspace
        angles = np.arange(scan.angle_min, scan.angle_max, scan.angle_increment)

        # Convert ranges to Cartesian coordinates
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        # Filter out values with an intensity lower than the threshold
        valid_points = intensities_norm > self.INTENSITY_THRESH
        # Filter out values that are out of range and on wrong side
        valid_points = self.SIDE * y > -0.2
        # Test cases limit x range to 0 < x < self.CLIP
        valid_points = np.logical_and(valid_points, x > 0.0)
        valid_points = np.logical_and(valid_points, x < self.CLIP + 4.0)

        # Use valid_points as a mask to filter out x and y values
        # Stack them into a 2D array to get coordinates
        sliced_coords = np.vstack((x[valid_points], y[valid_points]))

        # Return the coordinates sliced to a valid range
        return sliced_coords


    def parameters_callback(self, params):
        """
        DO NOT MODIFY THIS CALLBACK FUNCTION!

        This is used by the test cases to modify the parameters during testing.
        It's called whenever a parameter is set via 'ros2 param set'.
        """
        for param in params:
            if param.name == 'side':
                self.SIDE = param.value
                self.get_logger().info(f"Updated side to {self.SIDE}")
            elif param.name == 'velocity':
                self.VELOCITY = param.value
                self.get_logger().info(f"Updated velocity to {self.VELOCITY}")
            elif param.name == 'desired_distance':
                self.DESIRED_DISTANCE = param.value
                self.get_logger().info(f"Updated desired_distance to {self.DESIRED_DISTANCE}")
        return SetParametersResult(successful=True)


def main():
    rclpy.init()
    wall_follower = WallFollower()
    rclpy.spin(wall_follower)
    wall_follower.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
