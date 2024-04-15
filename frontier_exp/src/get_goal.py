#!/usr/bin/python3

import rospy
import rosbag
import tf2_ros
from geometry_msgs.msg import Pose, PoseStamped
from geometry_msgs.msg import Point32
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud

import numpy as np
from sklearn.cluster import KMeans
import time



class Frontier_Exp():

    def __init__(self) -> None:

        rospy.init_node("frontier_server")
        
        self.goal_pub = rospy.Publisher("/goal_pose", PoseStamped, queue_size=10)
        
        self.neighbourhood = 5
        self.n = np.int8((self.neighbourhood - 1)/2)
        self.candidate_match = 13
        self.cluster_number = 8
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # goal_gen = rospy.Service()
        # print("Frontier service armed!")
        # rospy.spin()
        while(not rospy.is_shutdown()):
            self.get_goal()



    def get_goal(self):
        
        self.map = rospy.wait_for_message("/map", OccupancyGrid, timeout=10)
        rospy.loginfo("Received the map!")

        # Getting the meta data from the map
        width = self.map.info.width
        height = self.map.info.height     

        map_data = np.empty((height, width), dtype=np.int8)
        candidates = []

        # Converting 1D array into 2D
        for i in range(height):
            base_index = i*width
            end_index = base_index + width
            map_data[i] = np.array(self.map.data[base_index:end_index])
        
        map_data[map_data > 0] = (self.neighbourhood**2) + 5

        # Getting cadidates
        for i in range(self.n, height - self.n - 1):
            for j in range(self.n, width - self.n - 1):
                # if self.is_candidate(map_data[i-self.n:i+self.n+1, j-self.n:j+self.n+1]):
                if np.sum(map_data[i-self.n:i+self.n+1, j-self.n:j+self.n+1]) == -self.candidate_match:
                    candidates.append([i,j])

        # Clustering the candidates
        labels, centroid = self.get_cluster(candidates)
        centroid = np.array(centroid)

        # Scoring and selecting a centroid
        scores = self.get_scores(labels, centroid)
        the_chosen_one = np.argmax(scores)
        
        # Selecting a random candidate from the chosen centroid
        rand_val = np.random.randint(0,np.shape(labels)[0] - 1)
        while(labels[rand_val] != the_chosen_one):
            rand_val = np.random.randint(0,np.shape(labels)[0] - 1)
        
        goal_pose = candidates[rand_val]

        self.test_output(goal_pose)
        pass



    def get_scores(self, labels, centroid):

        res = self.map.info.resolution
        phi_1 = 300    # Score multiplier for dist
        phi_2 = 0.7    # Score multiplier for mass    
        scores = np.empty((np.shape(centroid)[0]))
        score_accum = 0

        # Getting the TF between the map and the robot
        try:
            transform = self.tf_buffer.lookup_transform("map", "base_footprint", rospy.Time(0), rospy.Duration(5))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn("Transform lookup failed: {}".format(e))

        # Getting the robot indices
        robot_index = np.empty((2))
        robot_index[1] = np.int8((transform.transform.translation.x + self.map.info.origin.position.x)/res)
        robot_index[0] = np.int8((transform.transform.translation.y + self.map.info.origin.position.y)/res)

        for i in range((np.shape(centroid)[0])):
            dist = np.linalg.norm(centroid[i] - robot_index)
            mass = np.sum(labels == i)

            scores[i] = (phi_1/dist) + (phi_2*mass)
            score_accum = score_accum + scores[i]
        
        return(scores/score_accum)



    def get_cluster(self, point_dataset):
        for i in range(11):
            kmeans = KMeans(init='k-means++', n_clusters=self.cluster_number)
            kmeans.fit(point_dataset)
        return kmeans.labels_, kmeans.cluster_centers_



    def test_output(self, candidates):
        val = np.random.randint(0,62)
        # goal_point = candidates[val]
        goal_point = candidates
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = float(goal_point[1]*self.map.info.resolution + self.map.info.origin.position.x)
        goal.pose.position.y = float(goal_point[0]*self.map.info.resolution + self.map.info.origin.position.y)
        goal.pose.position.z = 0.0
        self.goal_pub.publish(goal)


    # def is_candidate(self, ker):

    #     # This function checks the eligibility of the neighbourhood to be a candidate
    #     # If it encounters an occupied cell, it returns False
    #     # If the number of unexplored points are less, it returns False

    #     # Counter for number of unexplored cells
    #     counter = 0

    #     # Convolution... sort of
    #     for i in range(self.neighbourhood):
    #         for j in range(self.neighbourhood):
    #             if ker[i,j] == 1:
    #                 return False
    #             elif ker[i,j] == 0:
    #                 counter = counter + 1

    #     if counter == self.candidate_match:
    #         return True
    #     else:
    #         return False

if __name__ == '__main__':
    Frontier_Exp()