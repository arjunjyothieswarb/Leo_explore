# MR_FinalProject
Repo for final project of mobile robotics

# Simulation Launch

## Instructions for setting up the environment:

Before launching the simulation, make sure to execute the following command in all terminals to set up the environment:

```bash
source devel/setup.bash
```

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
 
