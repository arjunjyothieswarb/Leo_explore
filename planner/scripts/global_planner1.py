import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from planner.srv import *
import tf2_ros

import numpy as np


class GlobalPlanner():

    def __init__(self) -> None:
        
        self.map = OccupancyGrid()
        low_res_map = OccupancyGrid()

        kernel_size = 3
        stride = 3

        rospy.init_node("GlobalPlanner")
        
        rospy.Subscriber("/map", OccupancyGrid, self.get_map)
        rospy.Publisher("/next_pose", PoseStamped, queue_size=10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        first_run = 1
        need_next_point = True

        while(not rospy.is_shutdown()):
            
            # if first_run:
            #     rospy.wait_for_service("get_goal")
            #     first_run = 0

            rospy.wait_for_service("get_goal")
            
            self.low_res_map = self.down_sample(kernel_size, stride)
            
            try:
                get_goal_pose = rospy.ServiceProxy("get_goal", frontier_goal)
                goal_pose = get_goal_pose(need_next_point)
                self.get_path(goal_pose)
            
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)

            pass
        pass



    def down_sample(self,kernel_size, stride):
        
        height = self.map.info.height
        width = self.map.info.width

        low_res_height = np.int((height - kernel_size + 1)/stride + 1)
        low_res_width = np.int((width - kernel_size + 1)/stride + 1)

        low_res_data = np.zeros(low_res_height, low_res_width)


        i2 = 0
        j2 = 0

        i = np.int((kernel_size + 1)/2)
        j = np.int((kernel_size + 1)/2)
        
        while(i < height - 1):
            while(j < width - 1):
                
                val = np.sum(self.map_data[i-kernel_size:i+kernel_size, j-kernel_size:j+kernel_size])
                
                if val > 0:
                    low_res_data[i2,j2] = 100
                elif val < 0:
                    low_res_data[i2,j2] = -1

                j = j + stride
                j2 = j2 + 1
                pass
            i = i + stride
            i2 = i2 + 1
            pass

        
        return low_res_data
    


    def get_path(self, goal_pose):
        current_pose = self.tf_buffer.lookup_transform("base_footprint", "map")
        
        pass
    
    
    
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