import numpy as np

class JointPositionTask:
    def __init__(self, joint_idx, value):
        self.joint_idx = joint_idx
        self.sigma_d = value  # desired joint value
        self.err = np.zeros((1, 1))
        self.J = np.zeros((1, 7))  
        self.error = []

    def update(self, robot):
        # Get current joint position
        sigma = robot.get_joint_position(self.joint_idx)

        # Compute error
        self.err = self.sigma_d - sigma
        self.error.append(np.abs(self.err))

        # Extract joint from the Jacobian 
        self.J[0, self.joint_idx + 3] = 1   # skip over base joints

    def get_jacobian(self):
        return self.J

    def get_error(self):
        return self.err

    def is_active(self):
        return True
