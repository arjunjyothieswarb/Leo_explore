#!/usr/bin/python3

import rospy
import rosbag
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import OccupancyGrid
import numpy as np
from sklearn.cluster import KMeans



class Frontier_Exp():

    def __init__(self) -> None:
        self.test_data()

        rospy.init_node("frontier_server")
        
        self.goal_pub = rospy.Publisher("/goal_pose", PoseStamped, queue_size=10)
        self.neighbourhood = 5
        self.n = np.int8((self.neighbourhood - 1)/2)
        self.candidate_match = 13
        self.cluster_number = 8
        # goal_gen = rospy.Service()
        # print("Frontier service armed!")
        # rospy.spin()
        self.get_goal()
        pass

    def test_data(self):
        # Function to be used during testing to feed map data
        bag = rosbag.Bag("/home/bhanu/leon_ws/data/start_map.bag")
        for topic, msg, t in bag.read_messages(['/map']):
            self.map = msg
            break
        pass

    def get_goal(self):
        
        # Uncomment the below line for getting message directly from topic
        # self.map = rospy.wait_for_message("/map", OccupancyGrid, timeout=10)

        # Getting the meta data from the map
        width = self.map.info.width
        height = self.map.info.height
        print(self.map.info.origin)

        map_data = np.empty((height, width), dtype=np.int8)
        candidates = []

        # print("width * height: ", width * height)

        # Converting 1D array into 2D
        for i in range(height):
            base_index = i*width
            end_index = base_index + width
            map_data[i] = np.array(self.map.data[base_index:end_index])
        
        # Getting cadidates
        for i in range(self.n, height - self.n - 1):
            for j in range(self.n, width - self.n - 1):
                if self.is_candidate(map_data[i-self.n:i+self.n+1, j-self.n:j+self.n+1]):
                    candidates.append([i,j])
        
        print(np.shape(candidates))
        print(width * height)

        labels, centroid = self.get_cluster(candidates)

        print(labels)
        print(len(labels))
        print(centroid)

    def get_cluster(self, point_dataset):
        for i in range(11):
            kmeans = KMeans(init='k-means++', n_clusters=self.cluster_number)
            kmeans.fit(point_dataset)
        return kmeans.labels_, kmeans.cluster_centers_

    def is_candidate(self, ker):

        # This function checks the eligibility of the neighbourhood to be a candidate
        # If it encounters an occupied cell, it returns False
        # If the number of unexplored points are less, it returns False

        # Counter for number of unexplored cells
        counter = 0

        # Convolution... sort of
        for i in range(self.neighbourhood):
            for j in range(self.neighbourhood):
                if ker[i,j] == 1:
                    return False
                elif ker[i,j] == 0:
                    counter = counter + 1

        if counter == self.candidate_match:
            return True
        else:
            return False

if __name__ == '__main__':
    Frontier_Exp()