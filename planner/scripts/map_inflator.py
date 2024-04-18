#!/usr/bin/python3

import rospy
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

import numpy as np

class Map_inflator():

    def __init__(self) -> None:

        rospy.init_node("Binary_inflator")

        # Getting the kernel size
        self.kernel_size = rospy.get_param("~kernel_size",5)
        self.n = np.int8((self.kernel_size - 1)/2)
        
        # Initializing the publisher and subscriber
        self.map_pub = rospy.Publisher("/binary_cost_map", OccupancyGrid, queue_size=10)
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.inflate_map)

        rospy.spin()

    def inflate_map(self, data):

        # Extracting map width and height from the message
        width = data.info.width
        height = data.info.height

        # Generating an empty arrays of the same size
        map_data = np.empty((height, width), dtype=np.int8)
        inflated_map_data = np.empty((height, width), dtype=np.int8)

        # Converting 1D array into 2D array
        for i in range(height):
            base_index = i*width
            end_index = base_index + width
            map_data[i] = np.array(data.data[base_index:end_index])

        inflated_map_data = np.array(map_data[:,:])

        # Inflating
        for i in range(self.n, height - self.n - 1):
            for j in range(self.n, width - self.n - 1):
                if map_data[i,j] > 0:
                    inflated_map_data[i-self.n:i+self.n+1, j-self.n:j+self.n+1] = 100


        inflated_map_data = inflated_map_data.flatten()
        data.data = tuple(inflated_map_data)

        data.header.stamp = rospy.Time.now()

        # Publishing the inflated map
        self.map_pub.publish(data)

if __name__ == '__main__':
    Map_inflator()