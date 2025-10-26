#!/usr/bin/env python3
import math
from dataclasses import dataclass
from typing import Optional
from __future__ import annotations
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from tf.broadcaster import TransformBroadcaster
from tf.transformations import quaternion_from_euler

# ---------------------------------------Helper dataclass                                                               
@dataclass
class Kinematics:
    v: float = 0.0  # linear (m/s)
    w: float = 0.0  # angular (rad/s)

# ---------------------------------------Main class
class DifferentialDriveOdometry:
    """ROS node that tracks pose by integrating wheel encoder rates."""                                                     

    def __init__(self) -> None:
        # Robot geometry — can be overridden via ROS parameters
        self._radius: float = rospy.get_param("~wheel_radius", 0.035)     # [m]
        self._baseline: float = rospy.get_param("~wheel_base", 0.23)       # [m]

        # Internal state (pose) and covariance
        self._x: np.ndarray = np.zeros((3, 1))                             # [x, y, yaw]
        self._P: np.ndarray = np.diag([0.04, 0.04, 0.04])                  # initial 3×3 cov

        # Process and measurement noise (tuned heuristically)
        self._Q: np.ndarray = np.diag([0.2 ** 2, 0.2 ** 2])                # wheel rate noise
        self._R: np.ndarray = np.diag([2.0, 2.0])                         

        # Time keeping
        self._last_stamp: Optional[rospy.Time] = None

        # Wheel speed cache (rad/s)
        self._left_rate: float = 0.0
        self._right_rate: float = 0.0

        # Publishes / Listeners
        self._odom_pub = rospy.Publisher("kobuki/odom", Odometry, queue_size=10)
        self._tf_br   = TransformBroadcaster()

        rospy.Subscriber("/velocities", JointState, self._on_joint_state)
        rospy.Subscriber("/goal_set",  Float64MultiArray, self._on_goal)

        # start integrating only after a goal is issued
        self._goal_received: bool = False

        rospy.loginfo("[DiffDriveOdom] Node initialised. Waiting for wheel data…")

    #------------------------------------------------------Callbacks                                                                 


    def _on_goal(self, _: Float64MultiArray) -> None:
        """Flag that a mission/goal has been issued."""
        self._goal_received = True
        rospy.loginfo_once("[DiffDriveOdom] Goal received, odometry live.")

    def _on_joint_state(self, msg: JointState) -> None:
        """Handle incoming wheel speed measurements and propagate pose."""
        # Optionally ignore data until we have a goal
        if not self._goal_received:
            return

        # Extract wheel angular velocities (rad/s)
        try:
            self._right_rate = msg.velocity[0]
            self._left_rate  = msg.velocity[1]
        except IndexError:
            rospy.logwarn_throttle(5.0, "[DiffDriveOdom] JointState missing velocity[0/1]")
            return

        # Compute time delta --------------------------------------------------
        stamp = msg.header.stamp
        if self._last_stamp is None:
            self._last_stamp = stamp
            return  # need two samples to compute Δt

        dt = (stamp - self._last_stamp).to_sec()
        self._last_stamp = stamp
        if dt <= 0.0:
            return

        # Kinematics ----------------------------------------------------------
        kin = self._wheel_rates_to_body(dt)

        # Pose integration ----------------------------------------------------
        self._x[0, 0] += math.cos(self._x[2, 0]) * kin.v * dt
        self._x[1, 0] += math.sin(self._x[2, 0]) * kin.v * dt
        self._x[2, 0] += kin.w * dt

        # Covariance prediction (discrete‑time EKF) ---------------------------
        A = np.array(
            [
                [1.0, 0.0, -math.sin(self._x[2, 0]) * kin.v * dt],
                [0.0, 1.0,  math.cos(self._x[2, 0]) * kin.v * dt],
                [0.0, 0.0, 1.0],
            ]
        )

        B = np.array(
            [
                [math.cos(self._x[2, 0]) * dt * 0.5, math.cos(self._x[2, 0]) * dt * 0.5],
                [math.sin(self._x[2, 0]) * dt * 0.5, math.sin(self._x[2, 0]) * dt * 0.5],
                [dt * self._radius / self._baseline, -dt * self._radius / self._baseline],
            ]
        )

        self._P = A @ self._P @ A.T + B @ self._Q @ B.T

        # Publish everything
        self._publish_odometry(stamp, kin, dt)

    #------------------------------------------------------Helper methods

    def _wheel_rates_to_body(self, dt: float) -> Kinematics:

        left_lin  = self._left_rate  * self._radius
        right_lin = self._right_rate * self._radius

        v = (left_lin + right_lin) / 2.0
        w = (right_lin - left_lin) / self._baseline
        return Kinematics(v=v, w=w)

    def _publish_odometry(self, stamp: rospy.Time, kin: Kinematics, dt: float) -> None:
        quat = quaternion_from_euler(0.0, 0.0, self._x[2, 0])

        # Build Odometry message
        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = "world_ned"
        odom.child_frame_id  = "turtlebot/kobuki/base_footprint"

        odom.pose.pose.position.x = self._x[0, 0]
        odom.pose.pose.position.y = self._x[1, 0]
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = quat[0]
        odom.pose.pose.orientation.y = quat[1]
        odom.pose.pose.orientation.z = quat[2]
        odom.pose.pose.orientation.w = quat[3]

        # Flatten covariance matrix into row‑major list (36 elements)
        Pflat = np.zeros((6, 6))
        Pflat[0:3, 0:3] = self._P
        odom.pose.covariance = Pflat.flatten().tolist()

        odom.twist.twist.linear.x  = kin.v
        odom.twist.twist.angular.z = kin.w
        self._odom_pub.publish(odom)

        # Broadcast TF 
        self._tf_br.sendTransform(
            (self._x[0, 0], self._x[1, 0], 0.0),
            quat,
            stamp,
            odom.child_frame_id,
            odom.header.frame_id,
        )


def main() -> None:
    rospy.init_node("differential_drive_odometry", anonymous=False)
    DifferentialDriveOdometry()
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
