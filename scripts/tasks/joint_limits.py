import numpy as np

class JointLimitsTask:
    def __init__(self, joint_idx, limits, thresholds):
        self.joint_idx = joint_idx
        self.limits = limits            # [q_min, q_max]
        self.thresholds = thresholds    # [alpha, sigma]
        self.err = np.zeros((1, 1))
        self.J = np.zeros((1, 7))
        self.active = 0                 # 0: inactive, 1: approaching lower limit, -1: approaching upper limit

    def update(self, robot):
        self.J[:] = 0
        self.J[0, self.joint_idx + 3] = 1   # skip over base joints

        # Get current joint position
        sigma = robot.get_joint_position(self.joint_idx)[0, 0]

        # Activate task if close to joint limits
        if self.active == 0:
            if sigma >= self.limits[1] - self.thresholds[0]:
                self.active = -1
            elif sigma <= self.limits[0] + self.thresholds[0]:
                self.active = 1
        # Deactivate task if moving back into safe zone
        elif self.active == -1 and sigma <= self.limits[1] - self.thresholds[1]:
            self.active = 0
        elif self.active == 1 and sigma >= self.limits[0] + self.thresholds[1]:
            self.active = 0

        # Apply error to keep robot within joint limits
        self.err[0, 0] = float(self.active)

    def get_jacobian(self):
        return self.J

    def get_error(self):
        return self.err

    def is_active(self):
        return self.active != 0
