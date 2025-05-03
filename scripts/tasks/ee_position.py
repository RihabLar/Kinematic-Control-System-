import numpy as np

class EndEffectorPositionTask:
    def __init__(self, value):
        self.sigma_d = value.reshape(3, 1)  # desired EE position
        self.err = np.zeros((3, 1))
        self.J = np.zeros((3, 7))
        self.error = []

    def update(self, robot):
        # Get current EE position
        sigma = robot.get_ee_position()

        # Compute error
        self.err = self.sigma_d - sigma
        self.error.append(np.linalg.norm(self.err))

        # Linear part of the Jacobian
        self.J = robot.get_jacobian()[0:3, :]

    def get_jacobian(self):
        return self.J

    def get_error(self):
        return self.err

    def is_active(self):
        return True
    
    def get_desired(self):
        return self.sigma_d
