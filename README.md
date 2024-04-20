# Leo Explore: The Reconnaissance Bot 

Welcome to Leo Explore: The Reconnaissance Bot  GitHub repository! Our project showcases an advanced autonomous navigation system designed for exploring unknown environments. Leveraging frontier exploration, adaptive path planning algorithms, and integration with the TurtleBot3 platform.

The planner can be run on simulation on Gazebo as well as the Turtlebot3 Burger platform.

## Instructions for setting up the environment:

Before launching the simulation, make sure to execute the following command in all terminals to set up the environment:

```bash
$ mkdir catkin_ws && cd catkin_ws
$ git clone https://github.com/arjunjyothieswarb/Leo_explore.git
$ mv Leo_explore src
$ catkin_make
```

```bash
$ source devel/setup.bash
```

Set Environment Variable:
```bash
$ export TURTLEBOT3_MODEL=burger
```
The source & export commands can be appended to the ~/.bashrc file to avoid redundant steps.

Install Packages: 
```bash
$ pip install scikit-learn
$ pip install scipy
$ sudo apt-get install ros-noetic-turtlebot3-gazebo
$ sudo apt-get install ros-noetic-turtlebot3-slam
$ sudo apt-get install ros-noetic-gmapping
$ sudo apt-get install ros-noetic-turtlebot3-teleop
```

Follow Simulation launch steps for Gazebo Simulation.

Follow Physical launch steps for testing it on the physical Turtlbot3 Burger platform.



## Simulation Launch

In Terminal:
```bash
$ export TURTLEBOT3_MODEL=burger
$ source devel/setup.bash
$ roslaunch planner tb3_gazebo_planner.launch

```

## Physical Launch

Create a ROS Master setup on your workstation with Turtlebot3 platform.

In Workstation Terminal:
```bash
$ roscore
```

In Turtlebot3 Terminal:
```bash
$ roslaunch turtlebot3_bringup turtlebot3_robot.launch
```


In Workstation Terminal:
```bash
$ export TURTLEBOT3_MODEL=burger
$ source devel/setup.bash
$ roslaunch planner tb3_remote.launch
```