#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point, Twist, PointStamped, Quaternion, PoseStamped
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry, Path
import tf2_ros
import numpy as np
# from mppi import MPPI
from planner.srv import global_path, global_pathResponse

class Unicycle():

    def __init__(self, v_min=0, v_max=0.2, w_min=-0.5, w_max=0.5):
        self.v_min= v_min
        self.v_max = v_max
        self.w_min = w_min
        self.w_max=w_max
        self.action_min = np.array([self.v_min, self.w_min])
        self.action_max = np.array([self.v_max, self.w_max])



    def step(
        self, current_state: np.ndarray, action: np.ndarray, dt: float = 0.1
    ) -> np.ndarray:
        """Move 1 timestep forward w/ kinematic model, x_{t+1} = f(x_t, u_t)"""
        # current_state = np.array([x, y, theta])
        # action = np.array([vx, vw])

        # clip the action to be within the control limits
        clipped_action = np.clip(
            action, np.array([self.v_min, self.w_min]), np.array([self.v_max, self.w_max])
        )

        current_state = current_state.reshape((-1, 3))
        clipped_action = clipped_action.reshape((-1, 2))
        next_state = np.empty_like(current_state)

        next_state[:, 0] = current_state[:, 0] + dt * clipped_action[
            :, 0
        ] * np.cos(current_state[:, 2])
        next_state[:, 1] = current_state[:, 1] + dt * clipped_action[
            :, 0
        ] * np.sin(current_state[:, 2])
        next_state[:, 2] = current_state[:, 2] + dt * clipped_action[:, 1]

        next_state = next_state.squeeze()

        return next_state

class MPPI:
    def __init__(
        self,
        motion_model=Unicycle(),
    ):
        """ Your implementation here """
        self.motion_model = motion_model

        self.N = 10
        self.lambda_ = 0.01
        # Each iteration updates state with control input for dt
        self.dt =0.02
        self.Tn=10#Total time = dt*Tn
        self.prev_control = None
        self.mean = [0.2, 0.0]
        self.std_dev = [0.1, np.pi/9]
        self.action_min = self.motion_model.action_min
        self.action_max = self.motion_model.action_max

    def score_rollouts(self, current_state, action):
        # Forward Simulation
        action = np.clip(action, a_min=self.action_min, a_max=self.action_max)
        # print(action)
        next_state = current_state
        for i in range(self.Tn):
          theta = current_state[2]
          control_mul = np.array([
              [np.cos(theta), np.sin(theta), 0],
              [0, 0, 1]
          ])
          next_state = (np.matmul(control_mul.T, action))*self.dt + current_state
          current_state = next_state
        # print(next_state)

        # Euclidean distance to goal & heading error

        norm_dist = np.sqrt((next_state[0]-self.goal_pos[0])**2 + (next_state[1]-self.goal_pos[1])**2)
        head_err = np.abs( next_state[2] - np.arctan2(self.goal_pos[1]-next_state[1], self.goal_pos[0]-next_state[0])) #Yaw - slope
        error = norm_dist

        return error, next_state



    def get_action(self, initial_state: np.ndarray, goal_pos: np.ndarray):
        """ Your implementation here """
        action=None
        self.goal_pos = goal_pos
        if self.prev_control is None:
          self.prev_control = [0.0, 0.0]

        # The perturbations
        del_controls = np.random.normal(self.mean, self.std_dev, size=(self.N, len(self.mean)))

        action = self.prev_control
        del_num=[0.0, 0.0]
        del_den= 0.0
        for delt in del_controls:
          cost, next_state = self.score_rollouts(initial_state, self.prev_control + delt)

          print(self.prev_control+delt, cost, delt)
          exp_term = np.exp(-1*cost/self.lambda_)
          del_num+= delt*exp_term
          del_den += exp_term

        del_num = del_num/del_den
        action+=del_num


        self.prev_control = action
        print(action)

        return np.clip(action, a_min=self.action_min, a_max=self.action_max)



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
