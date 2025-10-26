import numpy as np
import math


def compute_jacobian(q, base_yaw, base_distance, active_link):
    #Compute the 6×6 Jacobian matrix for the specified link.
    J_base_rotation = np.array([
        [-base_distance * np.sin(base_yaw) + (-np.sin(q[0, 0]) * np.sin(base_yaw) + 
         np.cos(q[0, 0]) * np.cos(base_yaw)) * (0.1588 * np.cos(q[2, 0]) + 0.056) - 
         0.142 * (-np.sin(q[0, 0]) * np.sin(base_yaw) + np.cos(q[0, 0]) * np.cos(base_yaw)) * 
         np.sin(q[1, 0])
        ],
        [base_distance * np.cos(base_yaw) + (np.sin(q[0, 0]) * np.cos(base_yaw) + 
         np.sin(base_yaw) * np.cos(q[0, 0])) * (0.1588 * np.cos(q[2, 0]) + 0.056) - 
         0.142 * (np.sin(q[0, 0]) * np.cos(base_yaw) + np.sin(base_yaw) * np.cos(q[0, 0])) * 
         np.sin(q[1, 0])
        ], 
        [0], [0], [0], [1]
    ])

    J_base_translation = np.array([
        [math.cos(base_yaw)], 
        [math.sin(base_yaw)], 
        [0], [0], [0], [0]
    ])
    J_joint1 = np.array([[(-np.sin(q[0, 0]) * np.sin(base_yaw) + np.cos(q[0, 0]) * np.cos(base_yaw)) * (0.1588 * np.cos(q[2, 0]) + 0.056) - 0.142 * (-np.sin(q[0, 0]) * np.sin(base_yaw) + np.cos(q[0, 0]) * np.cos(base_yaw) * np.sin(q[1, 0])) - 0.0132 * np.sin(q[0, 0]) * np.sin(base_yaw) + 0.0132 * np.cos(q[0, 0]) * np.cos(base_yaw)],
                        [(np.sin(q[0, 0]) * np.cos(base_yaw) + np.sin(base_yaw) * np.cos(q[0, 0])) * (0.1588 * np.cos(q[2, 0]) + 0.056) - 0.142 * (np.sin(q[0, 0]) * np.cos(base_yaw) + np.sin(base_yaw) * np.cos(q[0, 0])) * np.sin(q[1, 0]) + 0.0132 * np.sin(q[0, 0]) * np.cos(base_yaw) + 0.0132 * np.sin(base_yaw) * np.cos(q[0, 0])],
                        [0],
                        [0],
                        [0],
                        [1]])
    J_joint2 = np.array([
        [-0.142 * (np.sin(q[0, 0]) * np.cos(base_yaw) + np.sin(base_yaw) * np.cos(q[0, 0])) * np.cos(q[1, 0])],
        [-0.142 * (np.sin(q[0, 0]) * np.sin(base_yaw) - np.cos(q[0, 0]) * np.cos(base_yaw)) * np.cos(q[1, 0])],
        [0.142 * math.sin(q[1, 0])], 
        [0], [0], [0]
    ])

    J_joint3 = np.array([
        [-0.1588 * (np.sin(q[0, 0]) * np.cos(base_yaw) + np.sin(base_yaw) * np.cos(q[0, 0])) * np.sin(q[2, 0])],
        [-0.1588 * (np.sin(q[0, 0]) * np.sin(base_yaw) - np.cos(q[0, 0]) * np.cos(base_yaw)) * np.sin(q[2, 0])],
        [-0.1588 * math.cos(q[2, 0])], 
        [0], [0], [0]
    ])

    J_end_effector = np.array([[0], [0], [0], [0], [0], [1]])

    # Construct the full Jacobian matrix
    Jacobian = np.hstack((J_base_rotation, J_base_translation, J_joint1, J_joint2, J_joint3, J_end_effector))
    Jacobian[:, active_link:] = 0  # Zero out inactive links beyond current one

    return Jacobian
