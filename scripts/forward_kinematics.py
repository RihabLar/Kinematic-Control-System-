#!/usr/bin/env python3
import numpy as np

def compute_forward_kinematics(base_pose, q):
    """
    Inputs:
    q: [q1, q2, q3, q4] joint angles of the arm
    base_pose: (x_R, y_R, yaw_R) mobile base pose in the robots frame
    Outputs:
    EE position: (x_W, y_W, z_W) end effector position in the world frame
    """

    q1, q2, q3, q4 = q[0], q[1], q[2], q[3]
    x_R, y_R, yaw_R = base_pose

    # Constants
    d_base_horizontal = 13.2 / 1000.0 # horizontal distance from base to arm
    d_base_vertical = 108 / 1000.0    # vertical distance from base to arm
    L2 = 142 / 1000.0
    L3 = 158.8 / 1000.0
    d_grip = 56.7 / 1000.0            # 56.7 or 56.5?
    h_wrist = 72.2 / 1000.0
    offset = 50.7 / 1000.0
    total_angle = q1 + yaw_R

    # Compute horizontal (r) and vertical (z_E) components of planar reach
    r = d_base_horizontal + (L2 * np.sin(q2)) + (L3 * np.cos(q3)) + d_grip
    z_E = -d_base_vertical - (L2 * np.cos(q2)) - (L3 * np.sin(q3)) + h_wrist

    # Compute EE pose in world frame
    x_W = r * np.sin(total_angle) + (offset * np.cos(yaw_R)) + x_R
    y_W = -r * np.cos(total_angle) + (offset * np.sin(yaw_R)) + y_R
    z_W = z_E # differs from formula (z_W = z_E - d_base_horizontal)


    return np.array([x_W, y_W, z_W]).reshape(3, 1)
