#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker
from tf.transformations import euler_from_quaternion

from robot_model import RobotModel
from tasks.ee_position import EndEffectorPositionTask
from tasks.joint_position import JointPositionTask
from utils import DLS

class TaskPriorityController:
    def __init__(self):
        # Initilise node
        rospy.init_node('task_priority_node')
        self.rate = rospy.Rate(30)  # frequency of calling solve()

        # Robot state
        self.dof = 7  # [base_x, base_y, base_yaw, q1, q2, q3, q4]
        self.q = np.zeros((self.dof, 1))

        # Robot model
        self.robot = RobotModel()

        # Publishers and subscribers
        self.base_vel_pub = rospy.Publisher("/turtlebot/kobuki/commands/velocity", Twist, queue_size=1)
        self.arm_vel_pub = rospy.Publisher("/turtlebot/swiftpro/joint_velocity_controller/command", Float64MultiArray, queue_size=1)
        rospy.Subscriber("/turtlebot/joint_states", JointState, self.joint_state_callback)
        rospy.Subscriber("/turtlebot/kobuki/odom", Odometry, self.odom_callback)

        # Tasks
        self.tasks = []
        self.initialise_tasks()

        # Marker visualisation
        self.marker_pub = rospy.Publisher("/desired_ee_pos_marker", Marker, queue_size=1)

    def initialise_tasks(self):
        # Add first priority task
        self.tasks.append(EndEffectorPositionTask(value=np.array([0.3, 0.0, -0.2])))
        # Add second priority task
        #self.tasks.append(JointPositionTask(joint_idx=0, value=1))

    def joint_state_callback(self, msg):
        joint_names = {
            "turtlebot/swiftpro/joint1": 3,
            "turtlebot/swiftpro/joint2": 4,
            "turtlebot/swiftpro/joint3": 5,
            "turtlebot/swiftpro/joint4": 6
        }

        for i, name in enumerate(msg.name):
            if name in joint_names:
                self.q[joint_names[name]] = msg.position[i]

    def odom_callback(self, msg):
        # Get x, y, yaw from odometry
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.q[0] = position.x
        self.q[1] = position.y
        self.q[2] = yaw

    def run(self):
        while not rospy.is_shutdown():
            dq = self.solve()
            
            # Get base and arm velocities from dq
            vx = dq[0, 0]   # x linear velocity
            vy = dq[1, 0]   # y linear velocity
            wz = dq[2, 0]   # angular velocity about z
            q_dot = dq[3:].flatten()

            # Publish base velocities
            twist = Twist()
            twist.linear.x = vx
            twist.linear.y = vy
            twist.angular.z = wz
            self.base_vel_pub.publish(twist)

            # Publish arm velocities
            arm_msg = Float64MultiArray(data=q_dot[0:4].tolist())
            self.arm_vel_pub.publish(arm_msg)

            # Publish desired EE position marker
            desired_ee_pos = self.tasks[0].get_desired().flatten()
            self.publish_marker(desired_ee_pos)

            self.rate.sleep()

    def solve(self):
        dq = np.zeros((self.dof, 1))
        P = np.eye(self.dof)
        self.robot.update(self.q)

        for task in self.tasks:
            task.update(self.robot)

            if task.is_active():
                J = task.get_jacobian()
                J_bar = J @ P
                err = task.get_error()
                dq_task = DLS(J_bar, damping=0.1) @ (err - J @ dq)
                dq = dq + dq_task
                P = P - np.linalg.pinv(J_bar) @ J_bar

        return dq
    
    def publish_marker(self, position):
        marker = Marker()
        marker.header.frame_id = "world_ned"
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.scale.x = marker.scale.y = marker.scale.z = 0.05
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.marker_pub.publish(marker)


if __name__ == '__main__':
    controller = TaskPriorityController()
    controller.run()
