#!/usr/bin/env python3

import rospy
from nav_msgs.msg import OccupancyGrid
import cv2
import numpy as np

def map_callback(data):
    # Accessing occupancy grid map data
    grid_data = data.data
    width = data.info.width
    height = data.info.height
    resolution = data.info.resolution

    # Convert 1D grid data to 2D numpy array
    grid_array = np.array(grid_data).reshape((height, width))

    # Convert occupancy grid map values to grayscale image
    image = ((grid_array + 1) * 127.5).astype(np.uint8)


    # Display the image using OpenCV
    cv2.imshow('Occupancy Grid Map', image)
    cv2.waitKey(1)

def map_subscriber():
    rospy.init_node('map_subscriber', anonymous=True)
    rospy.Subscriber('/map', OccupancyGrid, map_callback)
    rospy.spin()

if __name__ == '__main__':
    map_subscriber()
