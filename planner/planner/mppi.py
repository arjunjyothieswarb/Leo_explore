import rospy
import numpy as np
import motion_model
from planner.scripts.motion_model import Unicycle

# class Unicycle():

#     def __init__(self, v_min=0, v_max=1, w_min=-2 * np.pi, w_max=2 * np.pi):
#         self.v_min= v_min
#         self.v_max = v_max
#         self.w_min = w_min
#         self.w_max=w_max
#         self.action_min = np.array([self.v_min, self.w_min])
#         self.action_max = np.array([self.v_max, self.w_max])



#     def step(
#         self, current_state: np.ndarray, action: np.ndarray, dt: float = 0.1
#     ) -> np.ndarray:
#         """Move 1 timestep forward w/ kinematic model, x_{t+1} = f(x_t, u_t)"""
#         # current_state = np.array([x, y, theta])
#         # action = np.array([vx, vw])

#         # clip the action to be within the control limits
#         clipped_action = np.clip(
#             action, np.array([self.v_min, self.w_min]), np.array([self.v_max, self.w_max])
#         )

#         current_state = current_state.reshape((-1, 3))
#         clipped_action = clipped_action.reshape((-1, 2))
#         next_state = np.empty_like(current_state)

#         next_state[:, 0] = current_state[:, 0] + dt * clipped_action[
#             :, 0
#         ] * np.cos(current_state[:, 2])
#         next_state[:, 1] = current_state[:, 1] + dt * clipped_action[
#             :, 0
#         ] * np.sin(current_state[:, 2])
#         next_state[:, 2] = current_state[:, 2] + dt * clipped_action[:, 1]

#         next_state = next_state.squeeze()

#         return next_state

class MPPI:
    def __init__(
        self,
        motion_model=Unicycle(),
    ):
        """ Your implementation here """
        self.motion_model = motion_model

        self.N = 40
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


        self.prev_control = np.clip(action, a_min=self.action_min, a_max=self.action_max)
        print(action)

        return np.clip(action, a_min=self.action_min, a_max=self.action_max)