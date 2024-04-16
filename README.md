# MR_FinalProject
Repo for final project of mobile robotics

# Simulation Launch

## Instructions for setting up the environment:

Before launching the simulation, make sure to execute the following command in all terminals to set up the environment:

```bash
source devel/setup.bash
```

Install Packages: 
```bash
$ pip install scikit-learn
$ pip install scipy
$ sudo apt-get install ros-noetic-turtlebot3-gazebo
$ sudo apt-get install ros-noetic-turtlebot3-slam
$ sudo apt-get install ros-noetic-gmapping
$ sudo apt-get install ros-noetic-turtlebot3-teleop
```

Set Environment Variable:
```bash
$ export TURTLEBOT3_MODEL=burger
``

Terminal 1:  
```bash
$ roslaunch planner tb3_gazebo_slam.launch 
```

Terminal 2:  
```bash
$ rosrun planner dummy_frontier_server.py 
```
 
Terminal 3:  
```bash
$ rosrun planner global_planner.py 
```
 
Terminal 4:  
```bash
$ rosrun planner local_planner.py 
```
 
