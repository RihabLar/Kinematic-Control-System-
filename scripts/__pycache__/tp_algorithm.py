#!/usr/bin/env python3

import rospy
import numpy as np
import math
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from tf.transformations import euler_from_quaternion

from robot_model import RobotModel
from utils import DLS_weighted, scale

# Task modules
from tasks.joint_limits import Jointlimits3D
from tasks.base_orientation import BaseOrientation3D
from tasks.ee_position import Position3D


class BasePositionTask:
    def __init__(self, goal):
        self.goal = np.array(goal).reshape(2, 1)
        self.active = True
        self.J = np.zeros((2, 6))
        self.err = np.zeros((2, 1))

    def update(self, robot):
        yaw = robot.getBasePose()[2, 0]
        J = np.zeros((2, 6))
        J[0:2, 1] = np.array([[np.cos(yaw)], [np.sin(yaw)]]).reshape(2)
        self.J = J
        pos = robot.getBasePose()[0:2]
        self.err = self.goal - pos

    def getError(self):
        return self.err

    def getJacobian(self):
        return self.J

    def is_active(self):
        return self.active


class TaskPriorityController:
    def __init__(self):
        rospy.init_node('task_priority_node')
        self.rate = rospy.Rate(30)

        self.origin = None
        self.dof = 6
        self.q = np.zeros((self.dof, 1))  # [yaw, d, q1, q2, q3, q4]

        self.robot = RobotModel(self.q[2:])

        self.max = 0.5
        self.max_a = self.max + 0.01 
        self.min = -0.5  
        self.min_a = self.min - 0.01 


        # Publishers
        self.base_vel_pub = rospy.Publisher("/turtlebot/kobuki/commands/velocity", Twist, queue_size=1)
        self.arm_vel_pub = rospy.Publisher("/turtlebot/swiftpro/joint_velocity_controller/command", Float64MultiArray, queue_size=1)
        self.goal_marker_pub = rospy.Publisher("/tp_goal_marker", Marker, queue_size=1)
        self.current_marker_pub = rospy.Publisher("/tp_current_marker", Marker, queue_size=1)

        # Subscribers
        rospy.Subscriber("/turtlebot/joint_states", JointState, self.joint_state_callback)
        rospy.Subscriber("/turtlebot/kobuki/odom", Odometry, self.odom_callback)

        self.tasks = []
        self.initialise_tasks()

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])

        if self.origin is None:
            self.origin = np.array([pos.x, pos.y, yaw])

        dx = pos.x - self.origin[0]
        dy = pos.y - self.origin[1]
        d_body_x = np.cos(yaw) * dx + np.sin(yaw) * dy

        self.q[0] = yaw
        self.q[1] = d_body_x

    def joint_state_callback(self, msg):
        joint_names = {
            "turtlebot/swiftpro/joint1": 2,
            "turtlebot/swiftpro/joint2": 3,
            "turtlebot/swiftpro/joint3": 4,
            "turtlebot/swiftpro/joint4": 5
        }
        for i, name in enumerate(msg.name):
            if name in joint_names:
                self.q[joint_names[name]] = msg.position[i]

    def initialise_tasks(self):
        self.W = np.diag([5.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        # Joint limits
        # Example using custom joint limit bounds for joint 1 and joint 2
        self.tasks.append(Jointlimits3D("joint1_limit", None, [self.min_a, self.min, self.max, self.max_a], 2))
        self.tasks.append(Jointlimits3D("joint2_limit", None, [self.min_a, self.min, self.max, self.max_a], 3))
        #self.tasks.append(Jointlimits3D("joint3_limit", None, [self.min_a, self.min, self.max, self.max_a], 4))
        #self.tasks.append(Jointlimits3D("joint4_limit", None, [self.min_a, self.min, self.max, self.max_a], 5))
        #self.tasks.append(Jointlimits3D("joint3_limit", None, [-1.8, -1.6, 1.6, 1.8], 4))
        #self.tasks.append(Jointlimits3D("joint4_limit", None, [-2.9, -2.5, 2.5, 2.9], 5))

        # Goal position
        desired_ee = np.array([0.6, 0.3, -0.3])
        yaw_target = np.arctan2(desired_ee[1], desired_ee[0])

        self.tasks.append(BaseOrientation3D("base_orient", yaw_target, 0))
        self.tasks.append(BasePositionTask(desired_ee[:2]))
        self.tasks.append(Position3D("ee_position", desired_ee.reshape(3, 1), 6))

    def solve(self):
        dq = np.zeros((self.dof, 1))
        P = np.eye(self.dof)
        self.robot.update(self.q)

        for i, task in enumerate(self.tasks):
            task.update(self.robot)
            print(f"[TASK {i}] {type(task).__name__} active? {task.is_active()}")

            if task.is_active():
                J = task.getJacobian()
                err = task.getError()
                J_bar = J @ P
                dq_task = DLS_weighted(J_bar, self.W, damping=0.1) @ (err - J @ dq)

                print(f"[TASK {i}] ‖err‖ = {np.linalg.norm(err):.3f}")
                print(f"[TASK {i}] dq_task = \n{dq_task.flatten()}")
                print(f"[TASK {i}] Jacobian = \n{J}")

                dq += dq_task
                P = P - np.linalg.pinv(J_bar) @ J_bar


        # Joint velocity limits
        vmax= 0.3
        vmin= -0.3
        # Enforce the velocity limits on the computed joint velocities
        for q in range(len(dq)):
            if dq[q] < vmin:
                dq = scale(dq, vmin, q)
            if dq[q] > vmax:
                dq = scale(dq, vmax, q)
        print(f"[TOTAL] dq = \n{dq.flatten()}")
        dq = dq.reshape(6, 1)


        return dq

    def run(self):
        while not rospy.is_shutdown():
            dq = self.solve()

            # Send base velocity
            twist = Twist()
            twist.angular.z = dq[0, 0]
            twist.linear.x = dq[1, 0]
            self.base_vel_pub.publish(twist)

            # Send arm velocity
            arm_msg = Float64MultiArray(data=dq[2:].flatten().tolist())
            self.arm_vel_pub.publish(arm_msg)

            self.publish_markers()
            self.rate.sleep()

    def publish_markers(self):
        for task in self.tasks:
            if isinstance(task, Position3D):
                goal = task.getDesired().flatten()
                self._publish_sphere(self.goal_marker_pub, goal, "ee_goal", (1.0, 0.0, 0.0))  # red

        current = self.robot.getEETransform()[0:3, 3].flatten()
        self._publish_sphere(self.current_marker_pub, current, "ee_current", (0.0, 1.0, 0.0))  # green

    def _publish_sphere(self, pub, pos, ns, rgb):
        m = Marker()
        m.header.frame_id = "world_ned"
        m.header.stamp = rospy.Time.now()
        m.ns = ns
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = pos[0]
        m.pose.position.y = pos[1]
        m.pose.position.z = pos[2]
        m.scale.x = m.scale.y = m.scale.z = 0.05
        m.color.r, m.color.g, m.color.b = rgb
        m.color.a = 1.0
        pub.publish(m)


if __name__ == "__main__":
    controller = TaskPriorityController()
    controller.run()
