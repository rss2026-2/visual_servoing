import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2

from vs_msgs.msg import ParkingError, ConeLocation


from safety_controller.safety_controller import SafetyController

class HomographyTester(Node):
    def __init__(self):
        super().__init__("homography_tester")
        
        # Declare test type parameter
        self.declare_parameter("test_type", "y")
        # Get test type parameter
        self.TEST_TYPE = self.get_parameter('test_type').get_parameter_value().string_value
        
        # Subscribe to the /scan to get lidar points to filter out, so we can get the detected cone lidar points
        # We are comparing the relative cone location against this to get homography error
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        
        # Subscribe to /relative_cone to get the relative cone location computed by the homography
        # We are comparing the filtered scan values against this to get homography error
        self.cone_sub = self.create_subscription(ConeLocation, "/relative_cone", self.cone_computed_callback, 1)
        
        # Publish homography error to visualize metric
        # Same attributes as ParkingError so just use that message type
        self.hom_error_pub = self.create_publisher(ParkingError, "/homography_error", 10)
        
        # Publish filtered laser scans to visualize in rviz
        self.test_pub = self.create_publisher(PointCloud2, "/filtered_scans_hom_test", 10)
        
        # Initialize cone scanned by lidar that we're comparing homography to
        self.cartesian_coords_avg = np.array([0, 0])

        self.get_logger().info("Homography Tester Initialized")
    
    def scan_callback(self, scan):
        
        lidar_msg = scan
        if lidar_msg is None: return
        # function to filter our laser data
        lidar_subset_calc = SafetyController.get_lidar_subset_calculator(
            lidar_msg.angle_min,
            lidar_msg.angle_max,
            lidar_msg.angle_increment,
            lidar_msg.ranges
        )
        
        if self.TEST_TYPE == "x":
            # Filter to only take lidar points of a narrow strip forward (moving cone in +x direction)
            cartesian_coords = lidar_subset_calc(
                angle_range = [-np.pi/16, np.pi/16],
                distance_range= [0, 3.9]
            )
        
        elif self.TEST_TYPE == "y":
            # Filter to only take lidar points close and at a wide range (moving cone in +y direction)
            cartesian_coords = lidar_subset_calc(
                angle_range = [-np.pi/4, np.pi/4],
                distance_range=[0, 3]
            )
            
        else:
            raise Exception("self.TEST_TYPE is neither 'x' or 'y'")
            
        # Publish these filtered scans as PointCloud2 data
        SafetyController.bag_filtered_scans(cartesian_coords, self.test_pub, self.get_clock().now().to_msg())
        
        # Take the average of x and average of y coordinates to find the mean xy coordinate
        self.cartesian_coords_avg = np.mean(cartesian_coords, axis=0)        
        
    def cone_computed_callback(self, msg):
        # Get the coords from the ConeLocation msg
        hom_coords = np.array([msg.x_pos, msg.y_pos])
        
        # Compare to the homography coordinate (different behavior for static vs. dynamic) to get homography error
        hom_error = hom_coords - self.cartesian_coords_avg
                
        # Populate ParkingError msg
        hom_error_msg = ParkingError()
        hom_error_msg.x_error = hom_error[0]
        hom_error_msg.y_error = hom_error[1]
        hom_error_msg.distance_error = np.sqrt(hom_error[0] ** 2 + hom_error[1] ** 2)
        
        # Publish ParkingError msg
        self.hom_error_pub.publish(hom_error_msg)

def main(args=None):
    rclpy.init(args=args)
    homography_tester = HomographyTester()
    rclpy.spin(homography_tester)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
