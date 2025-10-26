import numpy as np
import math

from src.forward_kinematics import compute_kinematics
from src.jacobian import compute_jacobian


class RobotModel:
    def __init__(self, q):
   
        self.q    = q
        self.revolute        = [True, False, True, True, True, True]
        self.dof             = len(self.revolute)
        self.base_pose       = np.zeros((3, 1))
        self.T_chain = np.zeros((4, 4))

    def update(self, dq, dt, mobile_pose):
 
        # 1) integrate arm joints
        x, y, yaw = mobile_pose

        # 2) update base pose
        self.base_pose[0, 0] = x
        self.base_pose[1, 0] = y
        self.base_pose[2, 0] = yaw    #¡¡¡¡¡

        self.q += (dq[2:, 0]).reshape(-1, 1) * dt

        # 3) build base‐to‐world transform   ¡¡¡¡¡
        T_bw = np.array([
            [math.cos(self.base_pose[2, 0]), -math.sin(self.base_pose[2, 0]), 0, self.base_pose[0, 0]],
            [math.sin(self.base_pose[2, 0]),  math.cos(self.base_pose[2, 0]), 0, self.base_pose[1, 0]],
            [0,                                0,                               1, 0],
            [0,                                0,                               0, 1]
        ])

        # 4) recompute all link transforms
        self.T_chain = compute_kinematics(self.q, T_bw)

    def getEETransform(self):
        # Get the EE transform (4×4 matrix)
        #print("T_chain", self.T_chain[-1])
        return self.T_chain[-1]

    def getEEJacobian(self, link):
        #Get the Jacobian for the specified link
        return compute_jacobian(self.q,
                                self.base_pose[2, 0],
                                self.base_pose[0, 0],
                                link)

    def getJointPos(self, link):
        #Get the position (angle) of the specified joint
        return self.q[link - 3]

    def getDOF(self):
        #Get total degrees of freedom.
        return self.dof

    def getLinkTransform(self, link):
        #Get the 4×4 transform of a specific link in the chain
        return self.T_chain[link - 1]
