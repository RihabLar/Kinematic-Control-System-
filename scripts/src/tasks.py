import numpy as np
import math
import rospy

class Task:
    def __init__(self, name, desired):

        self.name    = name
        self.sigma_d = desired
        self.active  = False
        self.J       = None
        self.err     = None

    def getDesired(self):
        return self.sigma_d

    def is_active(self):
        return self.active

    def getJacobian(self):
        return self.J

    def getError(self):
        return self.err

    def update(self, robot):
        raise NotImplementedError


class Position3D(Task):
    def __init__(self, name, desired, link):
        super().__init__(name, desired)
        self.link   = link
        self.J      = np.zeros((3, 6))
        self.err    = np.zeros((3, 1))
        self.active = True

    def update(self, robot):
        # Update the top‐3 rows of the Jacobian for position
        self.J = robot.getEEJacobian(self.link)[0:3] # return 3x6 jacobian up to link6
        # Current EE position
        pos = robot.getEETransform()[0:3, 3].reshape(3, 1)   # actual ee position
        # Error = desired − actual
        self.err = self.getDesired().reshape(3, 1) - pos 
        #print(f"[DEBUG] Position3D Task: Error = {self.err.T}")
        #print(f"[DEBUG] Jacobian J =\n{self.J}") 


class Orientation3D(Task):
    def __init__(self, name, desired, link):
        super().__init__(name, desired)
        self.link   = link
        self.J      = np.zeros((1, 6))
        self.err    = np.zeros((1,))
        self.active = True

    def update(self, robot):
        # Only the orientation row of the EE Jacobian
        self.J = robot.getEEJacobian(self.link)[-1].reshape(1,6)
        # Link transform, extract its planar orientation
        k     = robot.getLinkTransform(self.link)
        orien = np.arctan2(k[1, 0], k[0, 0])
        self.err = self.getDesired() - orien  

    
class BasePosition3D(Task):
    def __init__(self, name, desired, link):
        super().__init__(name, desired)
        self.link   = link
        self.J      = np.zeros((2, 6))  # Affecting [yaw_dot, forward_vel]
        self.err    = np.zeros((2, 1))
        self.active = True

    def update(self, robot):
        x, y, yaw = robot.base_pose.flatten()
        xd, yd = self.sigma_d.flatten()

        dx = xd - x
        dy = yd - y

        # Transform error into base frame
        err_local = np.array([
            [ np.cos(yaw)*dx + np.sin(yaw)*dy ],
            [-np.sin(yaw)*dx + np.cos(yaw)*dy ]
        ])

        self.err = err_local

        # Jacobian: how q affects base x/y in local frame
        self.J = np.array([
            [0, 1, 0, 0, 0, 0],  # affects q[1] (forward)
            [1, 0, 0, 0, 0, 0]   # affects q[0] (yaw)
        ])


class BaseOrientation3D(Task):
    def __init__(self, name, desired, link):
        super().__init__(name, desired)
        self.link   = link
        self.J      = np.zeros((1, 6))
        self.err    = 0.0
        self.active = True
    def update(self, robot):
        # The Jacobian for base yaw only affects q[0]
        self.J = np.zeros((1, 6))
        self.J[0, 0] = 1  # 1 DOF: yaw

        # Get current yaw of the base
        current_yaw = robot.base_pose[2]
        #rospy.loginfo(f"[DEBUG] BaseOrientation3D Task: Current yaw = {current_yaw}")

        # Compute wrapped error (difference between desired and current yaw)
        desired_yaw = self.getDesired()[0]
        #rospy.loginfo(f"[DEBUG] BaseOrientation3D Task: Desired yaw = {desired_yaw}")
        yaw_error = desired_yaw - current_yaw
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))  # Wrap to [-π, π]
        self.err = np.array([yaw_error])  # shape (1,)

    
class Configuration3D(Task):
    def __init__(self, name, desired, link):
        super().__init__(name, desired)
        self.link    = link
        self.J       = np.zeros((4, link))
        self.config  = np.zeros((4, 1))
        self.err     = np.zeros((4, 1))
        self.active  = True

    def update(self, robot):
        # Top 3 rows: position; last row: orientation
        self.J[0:3, :] = robot.getEEJacobian(self.link)[0:3]
        self.J[-1, :]  = robot.getEEJacobian(self.link)[-1].reshape(1, 6)

        trans = robot.getEETransform()
        ee_pose    = trans[0:3, 3].reshape(3, 1)
        current_orientation  = np.arctan2(trans[1, 0], trans[0, 0])
        self.config = np.vstack([ee_pose, current_orientation])
        self.err    = self.getDesired() - self.config  


class Jointlimits3D(Task):
    def __init__(self, name, desired, activation, link):
        super().__init__(name, desired)
        self.activation = activation  # [upper_off, upper_on, lower_on, lower_off]
        self.link       = link
        self.J          = np.zeros((1, 6))
        self.a          = 0
        self.err        = np.zeros((1,))
        self.active     = True

    def wrap_angle(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def update(self, robot):
        self.J = robot.getEEJacobian(self.link)[5, :].reshape(1, 6)
        link_T = robot.getLinkTransform(self.link)
        current_orientation  = np.arctan2(link_T[1, 2], link_T[0, 2])

        # Deactivate 
        if   self.a == 1 and current_orientation > self.activation[2]:
            self.a, self.active, self.err = 0, False, 0.0
        elif self.a == -1 and current_orientation < self.activation[0]:
            print('activated')
            self.a, self.active, self.err = 0, False, 0.0

        # Activate 
        if   self.a == 0 and current_orientation > self.activation[1]:
            print('un-activated')
            self.a, self.active, self.err = -1, True, -1.0
        elif self.a == 0 and current_orientation < self.activation[3]:
            self.a, self.active, self.err =  1, True,  1.0  

class JointPosition3D(Task):
    def __init__(self, name, desired, link):
        super().__init__(name, desired)
        self.link   = link
        self.J      = np.zeros((1, 6))
        self.err    = np.zeros((1,))
        self.active = True

    def update(self, robot):
        #self.J = robot.getEEJacobian(self.link)[5, :].reshape(1, 6)
        self.J = np.zeros((1, 6))
        self.J[0, self.link] = 1  # Directly control that joint

        #print(self.J)
        sigma    = robot.getJointPos(self.link)
        self.err = self.getDesired() - sigma  
        #print(self.err)