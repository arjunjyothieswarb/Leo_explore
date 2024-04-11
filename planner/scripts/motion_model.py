import numpy as np

class Unicycle():

    def __init__(self, v_min=0, v_max=1, w_min=-2 * np.pi, w_max=2 * np.pi):
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

