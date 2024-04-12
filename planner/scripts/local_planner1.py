#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point, Twist, PointStamped, Quaternion, PoseStamped
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry, Path, OccupancyGrid
import tf2_ros
import numpy as np
from planner.srv import global_path, global_pathResponse
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class Unicycle():

    def __init__(self, v_min=0.0, v_max=0.2, w_min=-2.0, w_max=2.0):
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
        static_map=None,
        map_metadata_obj=None,
        motion_model= Unicycle(),
        v_min=0, v_max=0.2, w_min=-np.pi, w_max=np.pi,
        num_rollouts = 50,
        num_steps = 10,
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
      W_samples = np.random.normal(0, 0.4, size=(self.num_rollouts, self.num_steps))
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
      v = 1
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

    def world_coordinates_to_map_indices(self, world_coordinates, resolution, origin):
      """
      Converts world coordinates to grid indices.
      Args:
          world_coordinates (tuple or list): World coordinates (x, y) in meters.
          resolution (float): Resolution of the map (cell size) in meters.
          origin (tuple or list): Origin of the map (x, y, theta) in meters and radians.
      Returns:
          tuple: Grid indices (i, j) in row and column format.
      """
      x, y = world_coordinates
      x_origin, y_origin, theta_origin = origin

      # Calculate relative position from origin of map
      dx = x - x_origin
      dy = y - y_origin

      # Rotate relative position based on origin theta
      x_rotated = dx * np.cos(-theta_origin) - dy * np.sin(-theta_origin)
      y_rotated = dx * np.sin(-theta_origin) + dy * np.cos(-theta_origin)

      # Convert to grid indices
      i = int(y_rotated / resolution)
      j = int(x_rotated / resolution)

      return i, j

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

      ## Quaternion to eulor conversion
      # yaw (z-axis rotation)
      siny_cosp = +2.0 * (q_w * q_z + q_x * q_y)
      cosy_cosp = +1.0 - 2.0 * (q_y * q_y + q_z * q_z)  
      yaw = np.arctan2(siny_cosp, cosy_cosp)
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
      inside_map = 0 <= i < height and 0 <= j < width

      return (i, j), inside_map

    def score_rollouts(self,initial_state: np.ndarray, goal_pos: np.ndarray, NxTplus1_states: np.ndarray):
      """
      Cost is calculated by calculating distance between the intermediat states to the goal state
      Returns: Normalized costs of each rollout
      """

      N, T_plus_1, _ = NxTplus1_states.shape
      D = self.calculate_distance(initial_state[:-1], goal_pos) # Distance form initial state to goal
      N_costs = []
      for n in range(N):
        cost = 0
        reached_goal = False
        hit_wall = 0
        for t in range(T_plus_1):
          if hit_wall!=0:
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
            row_index, col_index = map_indices
            if inside_map:
              hit_wall = self.map[row_index, col_index]
            else:
              hit_wall = 1

            #print("hit_wall :", hit_wall)
            if hit_wall != 0:
              cost += 10000
            elif d<=0.1:
              reached_goal = True
              cost += 0
            else:
              cost += d
        total_cost = cost/(self.num_steps)
        N_costs.append(total_cost)
      N_costs = np.array(N_costs)
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
        NxTplus1_states = self.simulting_action_seq(initial_state, np.expand_dims(updated_u_dash, axis=0))
        #self.plot_rollouts(initial_state, goal_pos, NxTplus1_states, env)                             #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


        '''
        Atharva Code: Ros Image Map check
        '''
        return updated_u_dash[0]


class ControlBot():
    def __init__(self,  amin=-0.2, amax=0.2, dt=0.1):
        # Initialize the ROS node
        rospy.init_node('local_planner', anonymous=True)
        cmd_vel_topic= '/cmd_vel'
        self.pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
        self.map_topic = "/map"
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
        # rospy.loginfo(f"{x}, {y}, {yaw}")

    def map_cb(self, data):

      grid_data = data.data
      width = data.info.width
      height = data.info.height
      resolution = data.info.resolution
      # Convert 1D grid data to 2D numpy array
      grid_array = np.array(grid_data).reshape((height, width))

      self.mppi.map = grid_array
      self.mppi.map_metadata_obj = data.info
      image = ((grid_array + 1) * 127.5).astype(np.uint8)
      # Create a publisher for the grayscale image. Replace 'grayscale_image_topic' with the desired topic name.
      image_pub = rospy.Publisher('grayscale_image_topic', Image, queue_size=10)

      # Initialize OpenCV bridge
      bridge = CvBridge()


      ros_image_msg = bridge.cv2_to_imgmsg(image, encoding="mono8")

      # Publish the grayscale image
      image_pub.publish(ros_image_msg)
      rospy.loginfo(data.header)




    def run_control(self):
      # rate = rospy.Rate()
      
      while not rospy.is_shutdown():
          try:
            rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
            # Get the latest transform from /map to /base_footprint
            transform = self.tf_buffer.lookup_transform('map', 'base_footprint', rospy.Time(0), rospy.Duration(1.0))
            
            # Callback to handle the transform
            self.transform_callback(transform)
            
            # rate.sleep()
          
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
      threshold_distance=0.02

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
