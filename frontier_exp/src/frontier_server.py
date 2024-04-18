#!/usr/bin/python3

import rospy
import rosbag
import tf2_ros
from geometry_msgs.msg import Pose, PoseStamped
from geometry_msgs.msg import Point32
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud
from planner.srv import frontier_goal, frontier_goalResponse

import numpy as np
import jax.numpy as jnp
import jax.scipy as jcp
from sklearn.cluster import KMeans
import time



class Frontier_Exp():

    def __init__(self) -> None:
        
        self.neighbourhood = 5
        self.n = np.int8((self.neighbourhood - 1)/2)
        self.ker = np.ones((self.neighbourhood, self.neighbourhood), dtype=jnp.int8)
        self.candidate_match = 12 #12
        self.cluster_number = 8
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)






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
        
        map_data[0][:] = 100
        map_data[:][0] = 100
        map_data[width-1][:] = 100
        map_data[:][height-1] = 100
        
        map_data[map_data > 0] = (self.neighbourhood**2) + 5

        # Getting cadidates
        map_data = jnp.array(map_data)
        map_data = jcp.signal.convolve(map_data, self.ker,'same')
        
        # cand_1 = map_data < -self.candidate_match
        # cand_2 = map_data > -((self.neighbourhood**2) - 5)
        cand_1 = map_data < -3
        cand_2 = map_data > -13

        # cand_1 = map_data.at[map_data<-self.candidate_match].set(True)
        # cand_2 = map_data.at[map_data>((self.neighbourhood**2) - self.candidate_match)].set(False)
        cand_map = jnp.logical_and(cand_1, cand_2)
        
        # map_data = np.array(map_data)
        # map_data[map_data < -self.candidate_match]# and map_data > -(self.neighbourhood**2 - self.candidate_match)] = 50
        
        candidates = jnp.argwhere(cand_map == True)
        # rospy.loginfo(f"Poses:{candidates}")
        print(candidates[:])

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
        rospy.loginfo(f"Index: {goal_pose}")

        goal_final= self.test_output(goal_pose)
        
        rospy.loginfo(f"Current Goal: {goal_final}")
        return goal_final



    def get_scores(self, labels, centroid):

        res = self.map.info.resolution
        phi_1 = 1000    # Score multiplier for dist
        phi_2 = 0.7    # Score multiplier for mass    
        scores = np.empty((np.shape(centroid)[0]))
        score_accum = 0

        # Getting the TF between the map and the robot
        try:
            transform = self.tf_buffer.lookup_transform("base_footprint", "map", rospy.Time(0), rospy.Duration(5))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn("Transform lookup failed: {}".format(e))

        # Getting the robot indices
        robot_index = np.empty((2))
        robot_index[1] = np.int16((transform.transform.translation.x - self.map.info.origin.position.x)/res)
        robot_index[0] = np.int16((transform.transform.translation.y - self.map.info.origin.position.y)/res)

        for i in range((np.shape(centroid)[0])):
            dist = np.linalg.norm(centroid[i] - robot_index)
            mass = np.sum(labels == i)

            scores[i] = (phi_1/dist) + (phi_2*mass)
            score_accum = score_accum + scores[i]
        
        return(scores/score_accum)



    def get_cluster(self, point_dataset):
        for i in range(5):
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
        
        return goal




def frontier_cb(request):
    if request.need_frontier:
        pose = PoseStamped()
        fs  = Frontier_Exp()
        pose=  fs.get_goal()
        return frontier_goalResponse(pose)
    else:
        return None


def get_goal():
    rospy.init_node("frontier_server")
    service = rospy.Service("frontier_goal", frontier_goal, frontier_cb)
    rospy.spin()


if __name__ == "__main__":
    get_goal()