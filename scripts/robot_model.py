import numpy as np
from forward_kinematics import compute_forward_kinematics
from jacobian import compute_jacobian

class RobotModel:
    def __init__(self):
        self.q = np.zeros((7, 1))
        self.ee_position = np.zeros((3, 1))
        self.jacobian = np.zeros((6, 7))

    def update(self, q):
        self.q = q
        base_pose = q[:3].flatten()
        joint_angles = q[3:].flatten()  # q1 to q4
        self.ee_position = compute_forward_kinematics(base_pose, joint_angles)
        self.jacobian = compute_jacobian(base_pose, joint_angles)

    def get_ee_position(self):
        return self.ee_position

    def get_jacobian(self):
        return self.jacobian

    def get_joint_position(self, joint_idx):
        return np.array([[self.q[joint_idx + 3, 0]]])  # skip over base joints
