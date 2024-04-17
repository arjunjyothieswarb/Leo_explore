import rospy
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from planner.srv import *
from planner.A_star import *
from tf.transformations import euler_from_quaternion
import tf2_ros

import numpy as np
import jax.numpy as jnp
from jax.lax import reduce_window

class GlobalPlanner():

    def __init__(self) -> None:
        
        self.map = OccupancyGrid()
        low_res_map = OccupancyGrid()

        kernel_size = 3
        stride = 3

        rospy.init_node("GlobalPlanner")
        
        rospy.Subscriber("/binary_cost_map", OccupancyGrid, self.get_map)
        self.pub_pose = rospy.Publisher("/next_pose", PoseStamped, queue_size=10)
        self.pub_path = rospy.Publisher("/goal_path", Path, queue_size=10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        first_run = 1
        need_next_point = True

        while(not rospy.is_shutdown()):

            rospy.wait_for_service("frontier_goal")
            
            # self.low_res_map = self.down_sample(kernel_size, stride)
            
            try:
                get_goal_pose = rospy.ServiceProxy("frontier_goal", frontier_goal)
                goal_pose = get_goal_pose(need_next_point)
                self.get_path(goal_pose.goal, kernel_size)
            
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)

            pass
        pass



    def down_sample(self,kernel_size, stride):
        
        map_data = np.array(self.map_data[:,:])
        map_data[map_data < 0] = 50
        map_data = jnp.array(map_data)

        low_res_data = np.array(reduce_window(map_data, jnp.max, (kernel_size, kernel_size), (stride,stride)))
        low_res_data = list(low_res_data)

        return low_res_data
    


    def get_path(self, goal_pose, kernel_size):
        
        path_msg = Path()
        current_pose = self.tf_buffer.lookup_transform("map", "base_footprint", rospy.Duration(0))
        
        pose_y = np.int16((current_pose.transform.translation.y - self.map.info.origin.position.y)/self.map.info.resolution)
        pose_x = np.int16((current_pose.transform.translation.x - self.map.info.origin.position.x)/self.map.info.resolution)
        # (pose_y, pose_x), _ = self.world_to_grid((current_pose.transform.translation.y, current_pose.transform.translation.x), self.map.info)

        pose_index = (pose_x, pose_y)
        # rospy.loginfo(self.map.info)
        
        goal_x = np.int16((goal_pose.pose.position.y - self.map.info.origin.position.y)/self.map.info.resolution)
        goal_y = np.int16((goal_pose.pose.position.x - self.map.info.origin.position.x)/self.map.info.resolution)
        goal_index = (goal_x, goal_y)
        start_index = (pose_x, pose_y)
        rospy.loginfo(f"Goal index: {goal_index}")
        rospy.loginfo(f"Start index: {start_index}")

        """The A* algorithm goes here"""
        # path = a_star_search(self.low_res_map, pose_index, goal_index)
        path = a_star_search(self.map_data, pose_index, goal_index)
        # path_x = path[0] * np.int8((self.kernel_size + 1)/2)
        # path_y = path[1] * np.int8((self.kernel_size + 1)/2)
        
        for (x,y) in path:
            temp_pose = PoseStamped()
            temp_pose.header.stamp = rospy.Time.now()
            temp_pose.header.frame_id = "map"
            temp_pose.pose.position.x = float(x*self.map.info.resolution + self.map.info.origin.position.x)
            temp_pose.pose.position.y = float(y*self.map.info.resolution + self.map.info.origin.position.y)
            temp_pose.pose.position.z = 0.0
            temp_pose.pose.orientation.w = 1.0
            rospy.loginfo(f"Path: {temp_pose.pose.position.x,temp_pose.pose.position.y}")
            path_msg.poses.append(temp_pose)
        
        rospy.loginfo(f"Start index: {len(path_msg.poses)}")

        
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"

        self.pub_path.publish(path_msg)
    
    
    
    def get_map(self, data):
        # Getting and storing the latest map
        self.map = data
        self.map_data = self.OneD_to_twoD(self.map.info.height, self.map.info.width)
    


    def OneD_to_twoD(self, height, width):
        # Converting 1D array into 2D
        
        map_data = np.empty((height, width), dtype=np.int8)
        
        for i in range(height):
            base_index = i*width
            end_index = base_index + width
            map_data[i] = np.array(self.map.data[base_index:end_index])
        
        return map_data
    
    def world_to_grid(self, world_coordinates, map_obj):
      """
      Converts world coordinates to grid indices and checks if the point is inside the map.
      
      Args:
          world_coordinates (tuple or list): World coordinates (x, y) in meters.
          map_obj (dict): Map object containing resolution, width, height, and origin information.
          
      Returns:
          tuple: Grid indices.
          bool: True if point is inside the map else False.
      """
      x, y = world_coordinates
      resolution = map_obj.resolution
      width = map_obj.width
      height = map_obj.height
      origin = map_obj.origin

      x_origin = origin.position.x
      y_origin = origin.position.y

      q_x = origin.orientation.x
      q_y = origin.orientation.y
      q_z = origin.orientation.z
      q_w = origin.orientation.w
      quaternion = (
      q_x,
      q_y,
      q_z,
      q_w
      )

      roll, pitch, yaw = euler_from_quaternion(quaternion)
      ## Quaternion to eulor conversion
      # yaw (z-axis rotation)
      # siny_cosp = +2.0 * (q_w * q_z + q_x * q_y)
      # cosy_cosp = +1.0 - 2.0 * (q_y * q_y + q_z * q_z)  
      # yaw = np.arctan2(siny_cosp, cosy_cosp)
      theta_origin = yaw

      # Calculate relative position from origin
      dx = x - x_origin
      dy = y - y_origin

      # Rotate relative position based on origin theta
      x_rotated = dx * np.cos(-theta_origin) - dy * np.sin(-theta_origin)
      y_rotated = dx * np.sin(-theta_origin) + dy * np.cos(-theta_origin)

      # Convert to grid indices
      i = int(y_rotated / resolution)
      j = int(x_rotated / resolution)

      # Check if point is inside the map
      inside_map = 0 <= j < height and 0 <= i < width

      return (j, i), inside_map
    

if __name__ == '__main__':
    GlobalPlanner()