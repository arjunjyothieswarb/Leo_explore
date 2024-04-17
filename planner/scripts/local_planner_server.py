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
        self.goal_pose_pub= rospy.Publisher("Waypt", PoseStamped, queue_size=10)
        self.waypt_goal =PoseStamped()
        self.get_path()


    # Service Request Function
    def get_path(self, ):
        # Request Server for goal pose
        rospy.loginfo("Path getting service")
        try:
            goal_srv = rospy.ServiceProxy("global_path", global_path)
            reached_goal =True
            response = goal_srv(reached_goal)
            self.global_path = response.path
            rospy.loginfo(response)
            self.waypt_track = WayptTracker(response.path)

        except rospy.ServiceException as e:
            rospy.loginfo("Service Exception: {}".format(e))        


    def transform_callback(self, transform):
    # Extracting translation and rotation from the transform
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        
        x = translation.x
        y = translation.y
        rospy.loginfo(rotation)
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
        goal, pose, done  = self.waypt_track.get_curr_waypt(self.state)
        if done:
          # Publishing zero vel
          cmd_vel= Twist()
          self.pub.publish(cmd_vel)
          self.get_path()
           

        action= self.mppi.get_action(self.state, goal)
        cmd_vel = Twist()
        cmd_vel.linear.x =action[0]
        cmd_vel.angular.z =action[1]
        self.pub.publish(cmd_vel)
        self.goal_pose_pub.publish(pose)

        rospy.loginfo(f"Action: {action}")  
        # rospy.loginfo(f"{x}, {y}, {yaw}")

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

      rospy.loginfo("grid_data shape"+ str(len(grid_data)))
      # dialation of map
      dilation_size = 4  
      dialated_map = self.dilate_obstacles(grid_array, dilation_size)

      self.mppi.map = dialated_map
      self.mppi.map_metadata_obj = data.info
      image = ((grid_array + 1) * 127.5).astype(np.uint8)

      unique_elements, counts = np.unique(data.data, return_counts=True)





    def run_control(self):
      # rate = rospy.Rate()
      
      while not rospy.is_shutdown():
          try:
            rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
            # Get the latest transform from /map to /base_footprint
            transform = self.tf_buffer.lookup_transform('map', 'base_footprint', rospy.Time(0), rospy.Duration(1.0))
            
            # Callback to handle the transform
            self.transform_callback(transform)
            
            # rate.sleep()
          
          except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("Transform lookup failed: {}".format(e))


class WayptTracker():
    def __init__(self, path:Path):
      self.path = path.poses
      self.num_poses = len(path.poses)
      self.i=0


    def get_curr_waypt(self, initial_state):
      # Update the waypoint
      pose = self.path[self.i]
      x = pose.pose.position.x
      y = pose.pose.position.y
      state = np.array([x,y])
      distance = np.linalg.norm(initial_state[:2] - state)
      threshold_distance=0.02

      if distance < threshold_distance:
          # Update the waypoint to a new position
          self.i+=1
      if self.i>=len(self.path):
        return np.array([x,y]), pose,True
      pose = self.path[self.i]
      x = pose.pose.position.x
      y = pose.pose.position.y



  
      return np.array([x,y]),pose, False



if __name__ == '__main__':
    try:
      # publising local path 
      #rospy.init_node('path_publisher_node', anonymous=True)
      path_pub = rospy.Publisher('/path_topic', Path, queue_size=10)
      cb = ControlBot()
      cb.run_control()
    except rospy.ROSInterruptException:
      pass
