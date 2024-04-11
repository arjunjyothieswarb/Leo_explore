import rospy
import numpy as np
from mppi import MPPI
from motion_model import Unicycle

class MPPIRacetrack(MPPI):
    def __init__(
        self,
        static_map,
        motion_model=Unicycle(),
    ):
        """ Your implementation here """
        super().__init__()
        self.static_map = static_map
        self.mean = [1.0, 0.0]
        self.std_dev = [0.20, np.pi/2]
        self.N=20
        self.i=0

    def score_rollouts(self, current_state, action):
        # Forward Simulation
        action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)
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





    def get_waypt_action(self, initial_state: np.ndarray, goal_pos: np.ndarray):
        """ Your implementation here """
        action=None
        self.goal_pos = goal_pos
        if self.prev_control is None:
          self.prev_control = [1.0, 0.0]

        # The perturbations
        del_controls = np.random.normal(self.mean, self.std_dev, size=(self.N, len(self.mean)))
        # fig = plt.figure(figsize=(10, 10))
        # plt.scatter(initial_state[0], initial_state[1])
        # plt.scatter(goal_pos[0], goal_pos[1])
        # print(del_controls)

        action = self.prev_control
        del_num=[0.0, 0.0]
        del_den= 0.0
        for delt in del_controls:
          cost, next_state = self.score_rollouts(initial_state, self.prev_control + delt)
          # self.plot_rollouts(initial_state, next_state, cost)
          # print(self.prev_control+delt, cost)
          exp_term = np.exp(-1*cost/self.lambda_)
          del_num+= delt*exp_term
          del_den += exp_term

        del_num = del_num/del_den
        action+=del_num

        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('Curve with Direction and Cost')
        # # plt.legend()
        # # plt.axis('equal')
        # plt.show()

        action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)
        self.prev_control = action
        print(action)

        return action

    waypts = np.array([[-3.0, 0.5], [-2.5, 3.0], [0.0, 4.5], [2.0, 4.0], [3.5, 2.0], [3.0,-2.0], [-0.5, -2.5]])
    
    def get_action(self, initial_state: np.array) -> np.array:
        """ Your implementation here """

        # Update the waypoint
        distance = np.linalg.norm(initial_state[:2] - self.waypts[self.i])
        threshold_distance=0.1
        if distance < threshold_distance:
            # Update the waypoint to a new position
            self.i+=1
            if self.i==7:
              self.i=0
        action = self.get_waypt_action(initial_state, self.waypts[self.i])
        print(self.waypts[self.i])

        return action