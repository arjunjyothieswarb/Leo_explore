import rospy
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from planner.srv import *
from planner.A_star import *
import tf2_ros

import numpy as np
import jax.numpy as jnp
from jax.lax import reduce_window

class GlobalPlanner():

    def __init__(self) -> None:
        
        self.map = OccupancyGrid()
        low_res_map = OccupancyGrid()

        self.GOT_MAP = False

        kernel_size = 3
        stride = 3

        rospy.init_node("GlobalPlanner")
        
        rospy.Subscriber("/binary_cost_map", OccupancyGrid, self.get_map)
        self.pub_pose = rospy.Publisher("/next_pose", PoseStamped, queue_size=10)
        self.pub_path = rospy.Publisher("/goal_path", Path, queue_size=10)
        self.pub_goal = rospy.Publisher("/incoming_goal", PoseStamped, queue_size=10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        first_run = 1
        need_next_point = True

        while(not rospy.is_shutdown()):

            rospy.wait_for_service("frontier_goal")
            
            # self.low_res_map = self.down_sample(kernel_size, stride)
            
            try:
                get_goal_pose = rospy.ServiceProxy("frontier_goal", frontier_goal)
                if not self.GOT_MAP:
                    continue
                goal_pose = get_goal_pose(need_next_point)
                self.pub_goal.publish(goal_pose.goal)
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
        
        goal_y = np.int16((goal_pose.pose.position.y - self.map.info.origin.position.y)/self.map.info.resolution)
        goal_x = np.int16((goal_pose.pose.position.x - self.map.info.origin.position.x)/self.map.info.resolution)
        goal_index = (goal_x, goal_y)
        rospy.loginfo(f"Grid val: {self.map_data[goal_x][goal_y]}")
        start_index = (pose_x, pose_y)
        rospy.loginfo(f"Goal index: {goal_index}")
        rospy.loginfo(f"Start index: {start_index}")

        """The A* algorithm goes here"""
        # path = a_star_search(self.low_res_map, pose_index, goal_index)
        if self.map_data[goal_x][goal_y] == 100:
            return
        path = a_star_search(self.map_data, pose_index, goal_index)
        
        
        for (x,y) in path:
            temp_pose = PoseStamped()
            temp_pose.header.stamp = rospy.Time.now()
            temp_pose.header.frame_id = "map"
            temp_pose.pose.position.x = float(x*self.map.info.resolution + self.map.info.origin.position.x)
            temp_pose.pose.position.y = float(y*self.map.info.resolution + self.map.info.origin.position.y)
            temp_pose.pose.position.z = 0.0
            temp_pose.pose.orientation.w = 1.0
            # rospy.loginfo(f"Path: {temp_pose.pose.position.x,temp_pose.pose.position.y}")
            path_msg.poses.append(temp_pose)
            if self.map_data[x][y] == 100:
                rospy.logwarn(f"Grid value is 100!")


        
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"

        self.pub_path.publish(path_msg)
    
    
    
    def get_map(self, data):
        # Getting and storing the latest map
        self.map = data
        self.map_data = self.OneD_to_twoD(self.map.info.height, self.map.info.width)
        self.GOT_MAP = True


    def OneD_to_twoD(self, height, width):
        # Converting 1D array into 2D
        
        map_data = np.empty((height, width), dtype=np.int8)
        
        for i in range(height):
            base_index = i*width
            end_index = base_index + width
            map_data[i] = np.array(self.map.data[base_index:end_index])
        
        return map_data
    

if __name__ == '__main__':
    GlobalPlanner()