#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point, Twist, PointStamped, Quaternion, PoseStamped
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry, Path
import tf2_ros
import numpy as np
from planner.mppi import MPPI, Unicycle
from planner.srv import global_path, global_pathResponse


class ControlBot():
    def __init__(self,  amin=-0.2, amax=0.2, dt=0.1):
        # Initialize the ROS node
        rospy.init_node('local_planner', anonymous=True)
        cmd_vel_topic= '/cmd_vel'
        self.pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
        self.sub_topic = "/odom"
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.mppi = MPPI()
        self.goal=None
        self.waypt_track = None
        self.get_path()

    # Service Request Function
    def get_path(self, ):
        # Request Server for goal pose
        rospy.loginfo("Path getting service")
        try:
            goal_srv = rospy.ServiceProxy("global_path", global_path)
            reached_goal =True
            response = goal_srv(reached_goal)
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
        roll, pitch, yaw = euler_from_quaternion(quaternion)
        self.state = np.array([x,y, yaw])
        goal = self.waypt_track.get_curr_waypt(self.state)
        action= self.mppi.get_action(self.state, goal)
        cmd_vel = Twist()
        cmd_vel.linear.x =action[0]
        cmd_vel.angular.z =action[1]
        self.pub.publish(cmd_vel)
        rospy.loginfo(f"Action: {action}")
        rospy.loginfo(f"Map pose: {x}, {y}, {yaw}")
        rospy.sleep(0.1)


    def run_control(self):
        while not rospy.is_shutdown():
            try:
                # Get the latest transform from /map to /base_footprint
                transform = self.tf_buffer.lookup_transform('map', 'base_footprint', rospy.Time(0), rospy.Duration(1.0))
                
                # Callback to handle the transform
                self.transform_callback(transform)
            
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
        threshold_distance=0.1

        if distance < threshold_distance:
            # Update the waypoint to a new position
            self.i+=1
        pose = self.path[self.i]
        x = pose.pose.position.x
        y = pose.pose.position.y

        return np.array([x,y])



if __name__ == '__main__':
    try:
        cb = ControlBot()
        cb.run_control()
    except rospy.ROSInterruptException:
        pass
