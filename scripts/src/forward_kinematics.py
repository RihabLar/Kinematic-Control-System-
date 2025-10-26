import numpy as np
import transforms3d.axangles as tf_ax
import math

def R_matrix(axis, angle):
    # Generate a 4×4 homogeneous rotation matrix
    M = np.eye(4)
    if axis == 'x':
        M[0:3, 0:3] = tf_ax.axangle2mat([1, 0, 0], angle)
    elif axis == 'z':
        M[0:3, 0:3] = tf_ax.axangle2mat([0, 0, 1], angle)
    return M

def T_matrix(vector):
    #Generate a 4×4 homogeneous translation matrix.
    M = np.eye(4)
    M[:3, 3] = vector
    return M

def base_to_link_transform():
    #Fixed transform from base footprint to swiftpro base.
    rot_angle = -math.pi / 2
    return np.array([
        [np.cos(rot_angle), -np.sin(rot_angle), 0, 0.051],
        [np.sin(rot_angle), np.cos(rot_angle), 0, 0],
        [0, 0, 1, -0.198],
        [0, 0, 0, 1]
    ])

def compute_kinematics(q, base_pose):
    #Calculate the transformation matrices from base to end-effector.
    list = [base_pose]

    transformation = [
        base_to_link_transform(),
        R_matrix('z', q[0, 0]) @ T_matrix([0.0132, 0, 0]) @ 
        R_matrix('x', -np.pi/2) @ T_matrix([0, 0.108, 0]),
        T_matrix([-0.142 * np.sin(q[1, 0]), 0.142 * np.cos(q[1, 0]), 0]),
        T_matrix([0.1588 * np.cos(q[2, 0]), 0.1588 * np.sin(q[2, 0]), 0]) @ 
        R_matrix('x', np.pi/2) @ T_matrix([0.056, 0, 0]),
        R_matrix('z', q[3, 0]) @ T_matrix([0, 0, 0.0722])
    ]

    for t in transformation:
        list.append(list[-1] @ t)

    return list
