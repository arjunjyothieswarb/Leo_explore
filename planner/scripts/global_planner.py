import rospy 
from geometry_msgs.msg import PoseStamped
from planner.srv import frontier_goal, frontier_goalResponse
from planner.srv import global_path, global_pathResponse
from nav_msgs.msg import Path


class DummyGlobalPlanner():
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('dummy_global_planner', anonymous=True)
        self.goal =PoseStamped()
        self.reach_goal = True

    def path_cb(self, request):
        rospy.loginfo(request)
        if request.reached_goal:
            rospy.loginfo(request)
            self.path = Path()
            self.path.header.frame_id = 'map'
            self.path.header.stamp = rospy.Time.now()
            # rospy.loginfo("1")
            self.get_goal()
            # rospy.loginfo("2")
            self.calculate_path()
            # rospy.loginfo("3")
            print(self.path)
            return global_pathResponse(self.path)
        else:
            return None

 
    def start_service(self):
        self.listen_service = rospy.Service("global_path", global_path, self.path_cb)
        rospy.spin()
        

    def get_goal(self):
        # Request Server for goal pose
        rospy.wait_for_service("frontier_goal")
        rospy.loginfo("Goal getting service")
        try:
            goal_srv = rospy.ServiceProxy("frontier_goal", frontier_goal)
            need_frontier =True
            response = goal_srv(need_frontier)
            rospy.loginfo(response)
            self.goal = response.goal
        except rospy.ServiceException as e:
            rospy.loginfo("Service Exception: {}".format(e))
    

    def calculate_path(self):
        # Just appending the goal_point as waypt
        self.path.poses.append(self.goal)


class GlobalPlanner(DummyGlobalPlanner):
    def __init__(self):
        super().__init__()

    def calculate_path(self):
        '''
        Add Path Calculation Logic
        use self.path storing path
        use self.map for latest map
        '''
        return super().calculate_path()



if __name__ == "__main__":
    dummy_on = rospy.get_param('dummy', True)
    if dummy_on:
        gp = DummyGlobalPlanner()
        gp.start_service()
    else:
        gp = GlobalPlanner()
        gp.start_service()