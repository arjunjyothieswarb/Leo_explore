import rospy 
import heapq
import numpy as np
from random import randint
from math import hypot
from geometry_msgs.msg import PoseStamped
from planner.srv import frontier_goal, frontier_goalResponse
from planner.srv import global_path, global_pathResponse
from nav_msgs.msg import Path


class GlobalPlanner():
    def __init__(self):
        rospy.init_node('global_planner', anonymous=True)
        self.goal =PoseStamped()
        self.reach_goal = False
        self.start = PoseStamped()
        self.resolution = 0.05 #meter/grid
        self.origin = PoseStamped() # Origin of world coordinates
        self.origin.pose.position.x = 0 #meter
        self.origin.pose.position.y = 0 #meter
        self.origin.pose.orientation.w = 1.0  # Assume no rotation
        # self.map_size = (384, 384)
        # self.map = self.generate_map()
        self.path = Path()
        self.path_publisher = rospy.Publisher("path_topic", Path, queue_size=10)
    
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

    # def generate_map(self):
    #     return np.zeros(self.map_size, dtype=int)
    
    def world_to_grid(self, world_coordinates):
        """Converts world coordinates to grid indices."""
        x, y = world_coordinates
        ix = int((x - self.origin.pose.position.x) / self.resolution)
        iy = int((y - self.origin.pose.position.y) / self.resolution)
        return (ix, iy)

    def heuristic(self, a, b):
        return hypot(b[0] - a[0], b[1] - a[1])

    def calculate_path(self):
        #use Astar algorithm
        start = self.world_to_grid((self.start.pose.position.x, self.start.pose.position.y))
        goal = self.world_to_grid((self.goal.pose.position.x, self.goal.pose.position.y))

        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            current = heapq.heappop(frontier)[1]

            if current == goal:
                break

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            next = (current[0] + dx, current[1] + dy)
            if 0 <= next[0] < self.map_size[0] and 0 <= next[1] < self.map_size[1] and self.map[next[0], next[1]] == 0:
                if dx != 0 and dy != 0:
                    new_cost = cost_so_far[current] + 1.414
                else:
                    new_cost = cost_so_far[current] + 1  
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(next, goal)
                    heapq.heappush(frontier, (priority, next))
                    came_from[next] = current

        path = self.reconstruct_path(came_from, start, goal)
        self.publish_path(path)
        return path  # Returning the path as a list of coordinates

    def reconstruct_path(self, came_from, start, goal):
        """Reconstruct the path from start to goal."""
        path = []
        current = goal
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path

    def publish_path(self, path):
        self.path = Path()
        self.path.header.frame_id = 'map'
        self.path.header.stamp = rospy.Time.now()
        for p in path:
            pose = PoseStamped()
            pose.header = self.path.header
            pose.pose.position.x = p[0] * self.resolution + self.origin.pose.position.x
            pose.pose.position.y = p[1] * self.resolution + self.origin.pose.position.y
            pose.pose.orientation.w = 1.0
            self.path.poses.append(pose)
        self.path_publisher.publish(self.path)



if __name__ == "__main__":
    gp = GlobalPlanner()
    gp.start_service()
