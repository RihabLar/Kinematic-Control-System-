#!/usr/bin/env python3
import numpy as np

def compute_jacobian(base_pose, q):
    """
    Inputs:
    q: [q1, q2, q3, q4] joint angles of the arm
    base_pose: (x_W, y_W, yaw_W) mobile base pose in the world frame
    Outputs:
    J: full Jacobian that maps joint velocities to end effector velocities
    """

    q1, q2, q3, q4 = q[0], q[1], q[2], q[3]
    x_W, y_W, yaw_W = base_pose

    d_base_horizontal = 13.2 / 1000.0 # horizontal distance from base to arm
    L2 = 142 / 1000.0
    L3 = 158.8 / 1000.0
    d_grip = 56.6 / 1000.0
    r = d_base_horizontal - (L2 * np.sin(q2)) + (L3 * np.cos(q3)) + d_grip
    offset = 50.7 / 1000.0
    total_angle = q1 + yaw_W

    # Trigonometric values
    sin_yaw = np.sin(yaw_W)
    cos_yaw = np.cos(yaw_W)
    sin_q2 = np.sin(q2)
    cos_q2 = np.cos(q2)
    sin_q3 = np.sin(q3)
    cos_q3 = np.cos(q3)
    sin_total_angle = np.sin(total_angle)
    cos_total_angle = np.cos(total_angle)

    # Full Jacobian
    J = np.zeros((6, 7))

    # Base translation (x, y)
    J[0, 0] = cos_yaw
    J[1, 0] = sin_yaw

    # Base rotation (yaw)
    J[0, 2] =  r * cos_total_angle - (offset * sin_yaw)
    J[1, 2] =  r * sin_total_angle + (offset * sin_yaw)

    # Joint 1
    J[0, 3] =  r * cos_total_angle
    J[1, 3] =  r * sin_total_angle

    # Joint 2
    J[0, 4] = -L2 * sin_total_angle * cos_q2
    J[1, 4] =  L2 * cos_total_angle * cos_q2
    J[2, 4] =  L2 * sin_q2

    # Joint 3
    J[0, 5] = -L3 * sin_total_angle * sin_q3
    J[1, 5] =  L3 * cos_total_angle * sin_q3
    J[2, 5] = -L3 * cos_q3

    # Joint 4
    J[5, 6] = 1.0

    return J
