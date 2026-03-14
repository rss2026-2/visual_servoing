#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np

from vs_msgs.msg import ConeLocation, ParkingError
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from std_msgs.msg import Header



class ParkingController(Node):
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """

    def __init__(self):
        super().__init__("parking_controller")

        self.declare_parameter("drive_topic")
        DRIVE_TOPIC = self.get_parameter("drive_topic").value  # set in launch file; different for simulator vs racecar

        self.drive_pub = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, 10)
        self.error_pub = self.create_publisher(ParkingError, "/parking_error", 10)

        self.create_subscription(
            ConeLocation, "/relative_cone", self.relative_cone_callback, 1)

        self.parking_distance = .75  # meters; try playing with this number!
        self.relative_x = 0
        self.relative_y = 0

        # added
        self.declare_parameter("car_length", 0.325)
        self.declare_parameter("max_steering_angle", 0.34)
        self.declare_parameter("velocity", 1.0)
        self.declare_parameter("lookahead", 0.8)
        self.declare_parameter("error_epsilon", 0.01)
        self.CAR_LENGTH = self.get_parameter('car_length').get_parameter_value().double_value
        self.MAX_STEERING_ANGLE = self.get_parameter('max_steering_angle').get_parameter_value().double_value
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        self.LOOKAHEAD = self.get_parameter('lookahead').get_parameter_value().double_value
        self.EPSILON = self.get_parameter('error_epsilon').get_parameter_value().double_value


        self.get_logger().info("Parking Controller Initialized")

    def relative_cone_callback(self, msg):
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos
        drive_cmd = AckermannDriveStamped()

        #################################

        # YOUR CODE HERE
        # Use relative position and your control law to set drive_cmd


        """We aren't aiming to give you a specific algorithm to run your controller, and we encourage you to play around. Try answering these questions:

        1. What should the robot do if the cone is far in front?
            Pure Persuit
        2. What should the robot do if it is too close?
            Reverse
        3. What if the robot isn't too close or far, but the cone isn't directly in front of the robot?
            reverse and steer towards, then go straight towards
        4. How can we keep the cone in frame when we are using our real camera?
            dont give super exageerated angles, don't go past"""

        self.get_logger().info(f'New Drive Command')

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'base_link'
        drive_cmd.header = header

        # Generate spline path
        path = self.generate_hermite_path(self.relative_x, self.relative_y)
        self.get_logger().info(f'Path: {path[0]}')

        # visualize path
        # VisualizationTools.plot_line(path[:,0], path[:,1], self.desired_publisher_)

        goal_dist = np.sqrt(self.relative_x**2 + self.relative_y**2)

        self.LOOKAHEAD = max(0.3, min(1.2, 0.5 * goal_dist))
        self.get_logger().info(f'{self.LOOKAHEAD=}')


        # choose lookahead target
        target_point = self.get_lookahead_point(path)

        pure_persuit_drive_cmd = self.update_control(target_point)

        drive_cmd.drive = pure_persuit_drive_cmd
        #################################

        self.drive_pub.publish(drive_cmd)
        self.error_publisher()

    def error_publisher(self):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """
        error_msg = ParkingError()

        #################################

        # YOUR CODE HERE
        # Populate error_msg with relative_x, relative_y, sqrt(x^2+y^2)
        error_msg.x_error = self.relative_x
        error_msg.y_error = self.relative_y
        error_msg.distance_error = np.sqrt(self.relative_x ** 2 + self.relative_y ** 2)

        #################################

        self.error_pub.publish(error_msg)

    def generate_hermite_path(self, cone_x, cone_y, tangent_strength=None, num_points=50):
        """
        Calculates a smooth path to a cone in the car's local frame.
        - cone_x, cone_y: Position of the cone relative to the car.
        - tangent_strength: Controls 'curviness'. Default is 1.5x the distance.
        """

        # 1. Start: Car is at (0,0) facing +X
        p0 = np.array([0.0, 0.0])

        # 2. Goal: Find the point parking_distance in front of the cone
        phi = np.arctan2(cone_y, cone_x) # Angle from car to cone
        p1 = np.array([
            cone_x - self.parking_distance * np.cos(phi),
            cone_y - self.parking_distance * np.sin(phi)
        ])

        # 3. Tangents: Direction car should be moving at start and end
        # Strength (L) affects how wide the turn is
        dist_to_goal = np.linalg.norm(p1 - p0)
        L = tangent_strength if tangent_strength else dist_to_goal * 1.5

        m0 = np.array([L, 0.0])              # Start tangent: straight forward
        m1 = np.array([L * np.cos(phi), L * np.sin(phi)]) # End tangent: facing cone
        # can also change this to be m1 = np.array([L, 0.0])

        # 4. Generate the Spline points
        t = np.linspace(0, 1, num_points)
        path = []
        for val in t:
            # Cubic Hermite Basis Functions
            h00 = 2*val**3 - 3*val**2 + 1
            h10 = val**3 - 2*val**2 + val
            h01 = -2*val**3 + 3*val**2
            h11 = val**3 - val**2

            # Calculate point on the curve
            point = h00*p0 + h10*m0 + h01*p1 + h11*m1
            path.append(point)

        return np.array(path)

    def get_lookahead_point(self, path):
        """
        Returns the first point on the path at least LOOKAHEAD distance away.
        """
        # for p in path:
        #     if np.linalg.norm(p) > self.LOOKAHEAD:
        #         return p

        # return path[-1]
        dists = np.linalg.norm(path, axis=1)
        closest_idx = np.argmin(dists)

        for i in range(closest_idx, len(path)):
            if np.linalg.norm(path[i]) > self.LOOKAHEAD:
                return path[i]

        return path[-1]

    def update_control(self, target_point):
        """
        Returns the ackerman drive command
        """
        drive = AckermannDrive()
        # in the case that the cone is behind the car, can also be modified for when we don't see the car
        if self.relative_x < 0:
            drive.speed = -0.5
            # steer toward the cone while reversing
            drive.steering_angle = float(np.clip(
                -np.sign(self.relative_y) * self.MAX_STEERING_ANGLE * 0.6,
                -self.MAX_STEERING_ANGLE,
                self.MAX_STEERING_ANGLE
            ))
            return drive


        # Check to see if we are too close
        goal_dist = np.sqrt(self.relative_x**2 + self.relative_y**2)

        if goal_dist < self.parking_distance and self.pointed_at_cone():
            # TODO add something here if we are too close but not pointed well...
            drive.speed = 0.0
            drive.steering_angle = 0.0
            return drive

        # calculate with the pure persuit
        new_steering_angle = self.compute_feedback_angle(target_point)

        # If the turn we have to make is too tight
        if abs(new_steering_angle) > self.MAX_STEERING_ANGLE * 0.9:
            drive.speed = -0.5
            reverse_angle = -0.5 * new_steering_angle
            drive.steering_angle = float(np.clip(reverse_angle,
                                     -self.MAX_STEERING_ANGLE,
                                     self.MAX_STEERING_ANGLE))

        else:
            drive.steering_angle = float(np.clip(new_steering_angle,
                                    -self.MAX_STEERING_ANGLE,
                                    self.MAX_STEERING_ANGLE))

            # slow down near goal
            dist = np.linalg.norm(target_point)

            if dist < 1.0:
                drive.speed = self.VELOCITY * 0.5
            else:
                drive.speed = self.VELOCITY

        return drive

    def compute_feedback_angle(self, target_point):
        """

        """
        lookahead_dist = np.linalg.norm(target_point)

        # pure pursuit steering law
        delta = np.arctan2(
            2 * self.CAR_LENGTH * target_point[1],
            lookahead_dist**2
        )

        # delta = np.clip(delta, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        # not clipping because we want to check potential reverse behavior
        return delta

    def pointed_at_cone(self):
        # Returns true if the relative y is within the alloted difference (rn set to 1cm)
        return abs(self.relative_y) < self.EPSILON

def main(args=None):
    rclpy.init(args=args)
    pc = ParkingController()
    rclpy.spin(pc)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
