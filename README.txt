# Kinematic Control System for a Mobile Manipulator

This project implements a **Task-Priority Kinematic Control System** for a **mobile manipulator** composed of:

* *Kobuki TurtleBot 2** – differential drive mobile base
* **uArm Swift Pro** – 4-DOF manipulator with a vacuum gripper
* **Sensors:** wheel encoders, RGB-D camera, and 2D LiDAR

Developed and tested in **ROS** and the **Stonefish simulator**, this system enables **autonomous pick-and-place operations** through coordinated control of the mobile base and manipulator.

---

## Overview

The control framework integrates multiple modules for perception, motion, and behavior management:

* **Task-Priority (TP)** redundancy resolution for coordinated base–arm motion
* **Behavior Trees** for flexible and modular task sequencing
* **Hybrid localization** using dead reckoning and **ArUco marker-based visual feedback**
* **Simulation-based testing** prior to physical implementation

---

## System Features

### Kinematic Control

* Implements **forward** and **inverse kinematics** for both the mobile base and manipulator
* Supports **Task-Priority control** with:

  * **Equality tasks:** end-effector position and orientation
  * **Inequality tasks:** joint limits, workspace constraints, and safety margins

### Pick-and-Place Pipeline

1. Move to the object location
2. Pick the object with the **vacuum gripper**
3. Transport it to the target location
4. Place the object
5. Return to the **home configuration**

### Navigation

* Combines **odometry-based motion** with **vision-based corrections** for robust navigation and positioning.

---

## Results

* Smooth and stable control of both base and arm in simulation
* **End-effector error:** within a few millimeters
* **Joint velocities:** maintained within safe limits
* **Partial real-world implementation** successfully tested

---

## Running the System

### Full Simulation (ArUco-Based Pick-and-Place)

**Step 1:** Launch the simulation environment

```bash
roslaunch intervention simulation.launch
```

**Step 2:** Run the behavior tree node

```bash
rosrun intervention behaviours_with_aruco.py
```

This will initialize the robot in the simulated environment and execute the complete **pick-and-place task** using **ArUco marker detection** and **task-priority control**.

---

### Dead Reckoning Task

**Step 1:** Launch the simulation environment

```bash
roslaunch intervention simulation.launch
```

**Step 2:** Run the Dead Reckoning node

```bash
rosrun intervention DeadReckoning.py
```

**Step 3:** Run the behavior tree node

```bash
rosrun intervention behaviours_with_aruco.py
```

---

## Dependencies

* [ROS (Robot Operating System)](https://www.ros.org/)
* [Stonefish Simulator](https://github.com/rapyuta-robotics/stonefish)
* Python 3.x
* `numpy`, `rospy`, `tf`, `cv2`, `aruco`

