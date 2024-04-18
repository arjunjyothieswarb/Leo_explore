import rospy
import numpy as np
import motion_model
from planner.motion_model import Unicycle
from tf.transformations import euler_from_quaternion
from scipy.ndimage import binary_dilation

class MPPI:
    def __init__(
        self,
        static_map=None,
        map_metadata_obj=None,
        motion_model= Unicycle(),
        v_min=0, v_max=0.2, w_min=-np.pi, w_max=np.pi,
        num_rollouts = 100,
        num_steps = 20,#50,
        lamda = 0.1,
        #env = gymnasium.Env                                        <<<<<<<<<<<<<<<<<<<<<<<<
    ):
        """ Your implementation here """
        self.motion_model = motion_model
        self.map = static_map
        self.map_metadata_obj = map_metadata_obj
        self.v_min = v_min
        self.v_max = v_max
        self.w_min = w_min
        self.w_max = w_max
        self.num_rollouts = num_rollouts
        self.num_steps = num_steps
        self.perturbations = None
        self.lamda = lamda
        #self.env = env   



    def generate_perturbations(self):
      '''
      Generates pertubations by randomly sampling form normal distrabution
      i/p : self
      o/p : np.ndarray of shape (self.num_rollouts,self.num_steps,2)
      '''
      V_samples = np.random.normal(0, 0.01, size=(self.num_rollouts, self.num_steps))
      V_scaled_samples = np.clip(V_samples, -1, 1)
      dv = V_scaled_samples
      W_samples = np.random.normal(0, 0.5, size=(self.num_rollouts, self.num_steps))
      W_scaled_samples = np.clip(W_samples, -1, 1)
      dw = W_scaled_samples * 2 * np.pi
      perturbations = []
      for n in range(self.num_rollouts):
        du_N = []
        for t in range(self.num_steps):
          du_N.append([dv[n,t], dw[n,t]])
        perturbations.append(du_N)
      self.perturbations = np.array(perturbations)

      # Plotting histograms of samples
      perturbations = np.array(self.perturbations)


    def open_loop_control_policy(self, init_state: np.ndarray, goal: np.ndarray, num_steps: int = 10):
      # The goal state is in global frame so we need to convert it to robot farame
      initial_x = init_state[0]
      initial_y = init_state[1]
      initial_theta = init_state[2]
      rotation_mat = [[ np.cos(initial_theta), np.sin(initial_theta)] ,
                      [-1 * np.sin(initial_theta), np.cos(initial_theta)]]
      g = goal.tolist()
      dif_initial_g = [g[0] - initial_x, g[1] - initial_y]
      x_relative , y_relative = np.matmul(rotation_mat, dif_initial_g)
      len_squ = (x_relative)**2 + (y_relative)**2

      # lets consider tangential velocity 1m/s
      v = 0.12
      if y_relative==0:
        w = 0.001
      else:
        r =  len_squ / (2* y_relative)
        w =  v / r
      control_sequence = []
      for i in range(num_steps):
        control_sequence.append(np.array([v,w]))
      return np.array(control_sequence)

    def simulting_action_seq(self, initial_state: np.ndarray, NxT_control_sequences):
      N, T, _ = NxT_control_sequences.shape
      NxTplus1_states = []
      for n in range(N):
        Tplus1_states = []
        state = initial_state
        Tplus1_states.append(state)
        for t in range(T):
          state = self.motion_model.step(state, NxT_control_sequences[n,t])
          Tplus1_states.append(state)
        NxTplus1_states.append(Tplus1_states)
      NxTplus1_states = np.array(NxTplus1_states)
      return NxTplus1_states



    def calculate_distance(self,point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance


    def world_to_grid(self, world_coordinates, map_obj):
      """
      Converts world coordinates to grid indices and checks if the point is inside the map.
      
      Args:
          world_coordinates (tuple or list): World coordinates (x, y) in meters.
          map_obj (dict): Map object containing resolution, width, height, and origin information.
          
      Returns:
          tuple: Grid indices.
          bool: True if point is inside the map else False.
      """
      x, y = world_coordinates
      resolution = map_obj.resolution
      width = map_obj.width
      height = map_obj.height
      origin = map_obj.origin

      x_origin = origin.position.x
      y_origin = origin.position.y

      q_x = origin.orientation.x
      q_y = origin.orientation.y
      q_z = origin.orientation.z
      q_w = origin.orientation.w
      quaternion = (
      q_x,
      q_y,
      q_z,
      q_w
      )

      roll, pitch, yaw = euler_from_quaternion(quaternion)
      ## Quaternion to eulor conversion
      # yaw (z-axis rotation)
      # siny_cosp = +2.0 * (q_w * q_z + q_x * q_y)
      # cosy_cosp = +1.0 - 2.0 * (q_y * q_y + q_z * q_z)  
      # yaw = np.arctan2(siny_cosp, cosy_cosp)
      theta_origin = yaw

      # Calculate relative position from origin
      dx = x - x_origin
      dy = y - y_origin

      # Rotate relative position based on origin theta
      x_rotated = dx * np.cos(-theta_origin) - dy * np.sin(-theta_origin)
      y_rotated = dx * np.sin(-theta_origin) + dy * np.cos(-theta_origin)

      # Convert to grid indices
      i = int(y_rotated / resolution)
      j = int(x_rotated / resolution)

      # Check if point is inside the map
      inside_map = 0 <= j < height and 0 <= i < width

      return (j, i), inside_map

    def score_rollouts(self,initial_state: np.ndarray, goal_pos: np.ndarray, NxTplus1_states: np.ndarray):
      """
      Cost is calculated by calculating distance between the intermediat states to the goal state
      Returns: Normalized costs of each rollout
      """

      N, T_plus_1, _ = NxTplus1_states.shape
      D = self.calculate_distance(initial_state[:-1], goal_pos) # Distance form initial state to goal
      N_costs = []
      hit_wall_count = 0
      self.test_positions = []
      for n in range(N):
        cost = 0
        reached_goal = False
        hit_wall = 0
        for t in range(T_plus_1):
          if hit_wall!= 0 and hit_wall != -1 :
            cost += 0
          elif reached_goal:
            cost += 0
          else:
            x = NxTplus1_states[n,t][0]
            y = NxTplus1_states[n,t][1]
            d = self.calculate_distance([x,y], goal_pos)

            ######
            # example
            world_coords = np.array([x, y])
            map_indices, inside_map = self.world_to_grid(world_coords, self.map_metadata_obj)
            self.test_positions.append(map_indices)
            row_index, col_index = map_indices
            hit_wall = self.map[col_index, row_index]


            #print("hit_wall :", hit_wall)
            if hit_wall != 0 and hit_wall != -1 :
              #rospy.loginfo("hit_wall:"+str(hit_wall))
              hit_wall_count += 1
              cost += 10000
            elif d<=0.1:
              reached_goal = True
              cost += 0
            else:
              cost += d
        total_cost = cost/(self.num_steps)
        N_costs.append(total_cost)
      N_costs = np.array(N_costs)
      # rospy.loginfo("Hit count: " + str(hit_wall_count))


      return N_costs

    def get_action(self, initial_state: np.array, goal_pos: np.ndarray) -> np.array:
        """ Your implementation here """

        ## Step 1 : Creating the random samples of N control perturbations seqences
        if self.perturbations is None:
          self.generate_perturbations()

        ## Step 2 : Pass control Sequences throught dynamic model
        # 2.1 : Generating primaray control using open loop control
        u_dash = self.open_loop_control_policy(init_state=initial_state, goal = goal_pos, num_steps= self.num_steps)

        # 2.2 : Adding perturbations and creating N control sequences
        NxT_control_sequences = np.copy(self.perturbations)
        for n in range(self.num_rollouts):
          for t in range(self.num_steps):
            NxT_control_sequences[n,t,0] = u_dash[t,0] + NxT_control_sequences[n,t,0]
            NxT_control_sequences[n,t,1] = u_dash[t,1] + NxT_control_sequences[n,t,1]

        # 2.3 : passing throught kinematics model
        """initial_state sequence_of_actions >> kinematics model >> states at each steps """
        NxTplus1_states = self.simulting_action_seq(initial_state, NxT_control_sequences)

        # 2.4 : Plotting Rollouts
        #self.plot_rollouts(initial_state, goal_pos, NxTplus1_states, env)                      #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        ## Step 3 : Scorring Rollouts
        """ states of each rollout at each step >> cost functin >> score of rollout """
        N_costs = self.score_rollouts(initial_state, goal_pos, NxTplus1_states)

        ## Step 4 : Updating weighted sum only for the initial step
        # 4.1 : Calcualting Weights
        exp_costs = np.exp(-1*N_costs/self.lamda)
        sum_exp_costs = np.sum(exp_costs)

        #rospy.loginfo(N_costs)
        N_weights = exp_costs/(sum_exp_costs + 0.000001)

        # 4.2 : Updating the initial control sequence
        NxT_perturbations = self.perturbations
        updated_u_dash = np.copy(u_dash)
        for t in range(self.num_steps):
          dv = 0
          dw = 0
          for n in range(self.num_rollouts):
            dv += NxT_perturbations[n,t][0] * N_weights[n]
            dw += NxT_perturbations[n,t][1] * N_weights[n]
          updated_u_dash[t][0] = updated_u_dash[t][0] + dv
          updated_u_dash[t][1] = updated_u_dash[t][1] + dw

        # checking the control limits
        updated_u_dash[:,0] = np.clip(updated_u_dash[:,0], self.v_min, self.v_max)
        updated_u_dash[:,1] = np.clip(updated_u_dash[:,1], self.w_min, self.w_max)

        # 4.3 Plot the updated trajectory
        # passing through simulatior or kinematics model
        NxTplus1_states_final = self.simulting_action_seq(initial_state, np.expand_dims(updated_u_dash, axis=0))
        #self.plot_rollouts(initial_state, goal_pos, NxTplus1_states, env)                             #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        


        return updated_u_dash[0]