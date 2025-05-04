import numpy as np
from forward_kinematics import compute_forward_kinematics

def compute_jacobian(base_pose, q):
    """
    Inputs:
    q: [q1, q2, q3, q4] joint angles of the arm (radians)
    Outputs:
    Jacobian matrix: 3x4 (rows: x, y, z; columns: joint velocities)
    """
    #--------------------------------- Manipulator Jacobian -------------------------#

    q1, q2, q3, q4 = q[:4]

    # Robot measurements in meters
    L2 = 142 / 1000.0          
    L3 = 158.8 / 1000.0       
    d_grip = 56.5 / 1000.0 
    d_base_vertical = 108 / 1000.0
    d_base_horizontal = 13.2 / 1000.0
    h_wrist = 72.2 / 1000.0

    # Trigonometric values
    sin_q1 = np.sin(q1)
    cos_q1 = np.cos(q1)
    sin_q2 = np.sin(q2)
    cos_q2 = np.cos(q2)
    sin_q3 = np.sin(q3)
    cos_q3 = np.cos(q3)

    # Compute horizontal (r) and vertical (z_E) components of planar reach
    r = d_base_horizontal - (L2 * sin_q2) + (L3 * cos_q3) + d_grip
    z_E = -d_base_vertical - (L2 * np.cos(q2)) - (L3 * np.sin(q3)) + h_wrist

    # Partial derivatives
    dr_dq2 = -L2 * cos_q2
    dr_dq3 = -L3 * sin_q3
    dzE_dq2 = L2 * sin_q2
    dzE_dq3 = -L3 * cos_q3

    # Jacobian terms for x-direction
    dx_dq1 = -r * sin_q1
    dx_dq2 = dr_dq2 * cos_q1
    dx_dq3 = dr_dq3 * cos_q1

    # Jacobian terms for y-direction
    dy_dq1 = r * cos_q1
    dy_dq2 = dr_dq2 * sin_q1
    dy_dq3 = dr_dq3 * sin_q1

    # Jacobian terms for z-direction
    dz_dq1 = 0.0
    dz_dq2 = dzE_dq2
    dz_dq3 = dzE_dq3

    # Build Jacobian
    J_arm = np.array([
        [dx_dq1, dx_dq2, dx_dq3, 0.0],
        [dy_dq1, dy_dq2, dy_dq3, 0.0],
        [dz_dq1, dz_dq2, dz_dq3, 0.0]
    ])

    #------------------------------------ Base Jacobian -----------------------------#

    # Get EE position relative to the base
    x_E = r * np.cos(q1)
    y_E = r * np.sin(q1)

    # Get yaw angle of base
    yaw_W = base_pose[2]
    sin_yaw = np.sin(yaw_W)
    cos_yaw = np.cos(yaw_W)

    # Jacobian terms
    dx_dyaw = -sin_yaw * x_E - cos_yaw * y_E
    dy_dyaw =  cos_yaw * x_E - sin_yaw * y_E
    dz_dyaw = 0.0

    # Build Jacobian
    J_base = np.array([
        [1.0, 0.0, dx_dyaw],
        [0.0, 1.0, dy_dyaw],
        [0.0, 0.0, dz_dyaw]
    ])
    
    #--------------------------------- Angular Jacobian -------------------------#
    # !!! Placeholder
    J_angular = np.zeros((3, 7))
    J_angular[2, 6] = 1.0

    #--------------------------------- Full Jacobian -------------------------#

    J_linear = np.hstack((J_base, J_arm)) 
    J_full = np.vstack((J_linear, J_angular))

    return J_full
