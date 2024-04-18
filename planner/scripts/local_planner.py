#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point, Twist, PointStamped, Quaternion, PoseStamped
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry, Path, OccupancyGrid
import tf2_ros
import numpy as np
from planner.mppi import MPPI, Unicycle
from planner.srv import global_path, global_pathResponse
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from scipy.ndimage import binary_dilation



class ControlBot():
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('local_planner', anonymous=True)
        cmd_vel_topic= '/cmd_vel'
        self.pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)


        self.map_topic = "/map"
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.mppi = MPPI()
        self.goal=None
        self.waypt_track = None
        # self.goal_pose_pub= rospy.Publisher("Waypt", PoseStamped, queue_size=10)
        self.waypt_goal =PoseStamped()
        _ = rospy.wait_for_message('/next_pose', PoseStamped, timeout=10)
        rospy.Subscriber("/next_pose", PoseStamped, self.goal_cb)  
        

    def goal_cb(self, data):
        # rospy.loginfo("Goal Callback")
        self.goal_pose = data.pose
        self.goal = np.array([self.goal_pose.position.x, self.goal_pose.position.y])



    def transform_callback(self, transform):
    # Extracting translation and rotation from the transform
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        
        x = translation.x
        y = translation.y
        # rospy.loginfo(rotation)
        quaternion = (
        rotation.x,
        rotation.y,
        rotation.z,
        rotation.w
        )
        self.x=x
        self.y=y

        roll, pitch, yaw = euler_from_quaternion(quaternion)
        self.yaw =yaw
        self.state = np.array([x,y, yaw])


    # Function to perform dilation
    def dilate_obstacles(self, map_array, dilation_size):
        # Convert obstacles (100) to binary (1) and free spaces to binary (0)
        binary_map = np.where(map_array == 100, 1, 0)
        
        # Perform binary dilation with specified size
        dilated_map = binary_dilation(binary_map, structure=np.ones((dilation_size, dilation_size)))
        
        # Convert back to original valuespath = Path()
        dilated_map = np.where(dilated_map, 100, map_array)
        
        return dilated_map

    def map_cb(self, data):

      grid_data = data.data
      width = data.info.width
      height = data.info.height
      resolution = data.info.resolution
      # Convert 1D grid data to 2D numpy array
      grid_array = np.array(grid_data).reshape((height, width))

      # rospy.loginfo("grid_data shape"+ str(len(grid_data)))
      # dialation of map
      dilation_size = 4  
      dialated_map = self.dilate_obstacles(grid_array, dilation_size)

      self.mppi.map = dialated_map
      self.mppi.map_metadata_obj = data.info


    def run_control(self):
      rate = rospy.Rate(10)
      
      while not rospy.is_shutdown():
          try:
            _ = rospy.wait_for_message('/next_pose', PoseStamped, timeout=10)
            rospy.Subscriber("/next_pose", PoseStamped, self.goal_cb)  

            rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
            # Get the latest transform from /map to /base_footprint
            transform = self.tf_buffer.lookup_transform('map', 'base_footprint', rospy.Time(0), rospy.Duration(1.0))
            
            # Callback to handle the transform
            self.transform_callback(transform)


            dist = np.linalg.norm(self.state[:2] - self.goal)
            while dist>0.05:
              action= self.mppi.get_action(self.state, self.goal)
              cmd_vel = Twist()
              cmd_vel.linear.x =action[0]
              cmd_vel.angular.z =action[1]
              self.pub.publish(cmd_vel)
              rospy.loginfo(f"Start: {self.state}") 
              rospy.loginfo(f"End:{self.goal}")  
              rospy.loginfo(f"Action: {action}") 
              rospy.loginfo(f"Distance: {dist}")  
              dist = np.linalg.norm(self.state[:2] - self.goal)
              transform = self.tf_buffer.lookup_transform('map', 'base_footprint', rospy.Time(0), rospy.Duration(1.0))
              
              # Callback to handle the transform
              self.transform_callback(transform)  

              rate.sleep()
            cmd_vel = Twist()
            self.pub.publish(cmd_vel)
          
          except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("Transform lookup failed: {}".format(e))





if __name__ == '__main__':
    try:
      # publising local path 
      #rospy.init_node('path_publisher_node', anonymous=True)
      # path_pub = rospy.Publisher('/path_topic', Path, queue_size=10)
      cb = ControlBot()
      cb.run_control()
    except rospy.ROSInterruptException:
      pass
