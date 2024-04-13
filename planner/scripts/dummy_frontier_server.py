import rospy 
from geometry_msgs.msg import PoseStamped
from planner.srv import frontier_goal, frontier_goalResponse

def frontier_cb(request):
    # Returns map origin as goal position
    if request.need_frontier:
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x =5
        pose.pose.position.y =5
        return frontier_goalResponse(pose)
    else:
        return None


def get_goal():
    rospy.init_node("dummy_frontier")
    service = rospy.Service("frontier_goal", frontier_goal, frontier_cb)
    rospy.spin()


if __name__ == "__main__":
    get_goal()