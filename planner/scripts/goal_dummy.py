#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped

def publish_goal_pose():
    rospy.init_node('goal_pose_publisher', anonymous=True)
    pub = rospy.Publisher('/goal_pose', PoseStamped, queue_size=10)
    rate = rospy.Rate(1)  # 1hz

    while not rospy.is_shutdown():
        # Create a PoseStamped message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"  # Set the frame id to whatever frame the pose is in
        pose_msg.pose.position.x = -1.5  # Set the x coordinate of the pose
        pose_msg.pose.position.y = 0.0  # Set the y coordinate of the pose
        pose_msg.pose.position.z = 0.0  # Set the z coordinate of the pose
        pose_msg.pose.orientation.x = 0.0  # Set the x component of the orientation
        pose_msg.pose.orientation.y = 0.0  # Set the y component of the orientation
        pose_msg.pose.orientation.z = 0.0  # Set the z component of the orientation
        pose_msg.pose.orientation.w = 1.0  # Set the w component of the orientation

        # Publish the message
        pub.publish(pose_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_goal_pose()
    except rospy.ROSInterruptException:
        pass
