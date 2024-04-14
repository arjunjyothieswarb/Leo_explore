import rospy 
import heapq
import numpy as np
from random import randint
from math import hypot
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
            rospy.loginfo("1")
            self.get_goal()
            rospy.loginfo("2")
            self.calculate_path()
            rospy.loginfo("3")
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
        self.reach_goal = False
        self.start = PoseStamped()
        self.start.pose.position.x = 3  
        self.start.pose.position.y = 3
        self.map_size = (100, 100)
        self.map = self.generate_map()
        self.path = Path()

    

    def generate_map(self):
        map = np.zeros(self.map_size, dtype=int)
        return map

    def heuristic(self, a, b):
        return hypot(b.pose.position.x - a.pose.position.x, b.pose.position.y - a.pose.position.y)

    def calculate_path(self):
        #use Astar algorithm
        start = (int(self.start.pose.position.x), int(self.start.pose.position.y))
        goal = (int(self.goal.pose.position.x), int(self.goal.pose.position.y))

        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            current = heapq.heappop(frontier)[1]

            if current == goal:
                break

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-connected grid
                next = (current[0] + dx, current[1] + dy)
                if 0 <= next[0] < self.map_size[0] and 0 <= next[1] < self.map_size[1] and self.map[next[0], next[1]] == 0:
                    new_cost = cost_so_far[current] + 1
                    if next not in cost_so_far or new_cost < cost_so_far[next]:
                        cost_so_far[next] = new_cost
                        priority = new_cost + self.heuristic(next, goal)
                        heapq.heappush(frontier, (priority, next))
                        came_from[next] = current

        # Reconstruct path
        path = []
        current = goal
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()  # optional
        self.publish_path(path)

    def publish_path(self, path):
        self.path = Path()
        self.path.header.frame_id = 'map'
        self.path.header.stamp = rospy.Time.now()
        for p in path:
            pose = PoseStamped()
            pose.header = self.path.header
            pose.pose.position.x = p[0]
            pose.pose.position.y = p[1]
            pose.pose.orientation.w = 1.0
            self.path.poses.append(pose)
        self.path_publisher.publish(self.path)



if __name__ == "__main__":
    dummy_on = rospy.get_param('dummy', True)
    if dummy_on:
        gp = DummyGlobalPlanner()
        gp.start_service()
    else:
        gp = GlobalPlanner()