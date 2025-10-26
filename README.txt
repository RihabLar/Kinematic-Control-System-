#Kinematic Control System for a Mobile Manipulator

This project implements a **task-priority kinematic control system** for a mobile manipulator composed of:
- **Kobuki TurtleBot 2** (differential drive base)
- **uArm Swift Pro** (4-DOF manipulator with vacuum gripper)
- Sensors: wheel encoders, RGB-D camera, 2D LiDAR

The system was developed and tested in **ROS** and the **Stonefish simulator**, with the goal of enabling autonomous **pick-and-place tasks**.

---

## Overview
- Uses the **Task-Priority (TP) redundancy resolution algorithm** to coordinate base and arm motion.
- Integrates **behavior trees** for task sequencing.
- Supports **dead reckoning** and **ArUco marker-based visual feedback** for navigation and object localization.
- Includes **simulation-based testing** for validation before real-world deployment.

---

## Features
- Forward and inverse kinematics for the mobile manipulator.
- Task-priority control with both **equality tasks** (end-effector position/orientation) and **inequality tasks** (joint limits, safety).
- Pick-and-place pipeline:
  - Move to object
  - Pick with vacuum gripper
  - Transport
  - Place at target
  - Return home
- Hybrid navigation: odometry + vision-based corrections.

---

##  Results
- Simulation showed accurate and smooth control of both base and arm.
- End-effector errors reduced to a few millimeters.
- Joint velocities remained within safe limits.
- Real-world implementation was partially completed.

---

To launch the full simulation with the ArUco-based pick-and-place task:

1. Start the simulation environment

   roslaunch intervention simulation.launch

2. Run the behavior tree node

    rosrun intervention behaviours_with_aruco.py

This will initialize the robot in the simulated environment and execute the full pick-and-place task using ArUco marker detection and the task-priority control system.

To run DeadReckoning Task: 

1. Start the simulation environment

   roslaunch intervention simulation.launch

2. Run DeadReckoning node

    rosrun intervention DeadReckoning.py

3. Run the behavior tree node

    rosrun intervention behaviours_with_aruco.py

