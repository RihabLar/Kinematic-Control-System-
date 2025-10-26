#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import math
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray

import tf2_ros
import tf2_geometry_msgs
from tf.transformations import quaternion_from_matrix, euler_from_quaternion


class ArucoDetector:
    def __init__(self):
        rospy.init_node("aruco_detector")

        # TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publishers
        self.pose_pub = rospy.Publisher("/aruco_position", PoseStamped, queue_size=10)
        self.goal_pub = rospy.Publisher("/goal_set", Float64MultiArray, queue_size=10)

        # Subscriber to camera image
        self.image_sub = rospy.Subscriber(
            "/turtlebot/kobuki/realsense/color/image_color",
            Image,
            self.image_callback
        )

        # CvBridge
        self.bridge = CvBridge()

        # ArUco configuration
        self.marker_length = 0.05  # meters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

        # Camera intrinsics
        self.camera_matrix = np.array([
            [1396.8086675255468, 0.0, 960.0],
            [0.0, 1396.8086675255468, 540.0],
            [0.0, 0.0, 1.0]
        ])
        self.dist_coeffs = np.zeros((5,))

        rospy.loginfo("ArucoDetector with yaw initialized")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge error: {e}")
            return

        corners, ids, _ = cv2.aruco.detectMarkers(cv_image, self.aruco_dict)

        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners[0], self.marker_length,
                self.camera_matrix, self.dist_coeffs)

            rvec = rvecs[0][0]
            tvec = tvecs[0][0]

            # Build transformation matrix (marker in camera frame)
            marker_cam = np.eye(4)
            marker_cam[:3, :3] = cv2.Rodrigues(rvec)[0]
            marker_cam[:3, 3] = tvec

            # Convert to quaternion
            quat = quaternion_from_matrix(marker_cam)

            # Create PoseStamped in camera frame
            pose_cam = PoseStamped()
            pose_cam.header.stamp = msg.header.stamp
            pose_cam.header.frame_id = msg.header.frame_id  # typically "turtlebot/kobuki/realsense_color"
            pose_cam.pose.position.x = tvec[0]
            pose_cam.pose.position.y = tvec[1]
            pose_cam.pose.position.z = tvec[2]
            pose_cam.pose.orientation.x = quat[0]
            pose_cam.pose.orientation.y = quat[1]
            pose_cam.pose.orientation.z = quat[2]
            pose_cam.pose.orientation.w = quat[3]

            try:
                tfm = self.tf_buffer.lookup_transform(
                    "world_ned", msg.header.frame_id,
                    rospy.Time(0), rospy.Duration(1.0))

                pose_world = tf2_geometry_msgs.do_transform_pose(pose_cam, tfm)
                self.pose_pub.publish(pose_world)

                # Extract yaw from quaternion
                q = pose_world.pose.orientation
                _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

                # Publish x, y, z, yaw to /goal_set
                goal_msg = Float64MultiArray()
                goal_msg.data = [
                    pose_world.pose.position.x,
                    pose_world.pose.position.y,
                    pose_world.pose.position.z,
                    yaw
                ]
                #self.goal_pub.publish(goal_msg)
                rospy.set_param('aruco_pose', [
                        pose_world.pose.position.x,
                        pose_world.pose.position.y,
                        pose_world.pose.position.z,
                        yaw
                    ])

                '''rospy.loginfo("Pose → x: %.2f, y: %.2f, z: %.2f, yaw: %.1frad",
                              goal_msg.data[0],
                              goal_msg.data[1],
                              goal_msg.data[2],
                              yaw)'''

            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"TF lookup failed: {e}")

            # Draw detected marker and axes
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
            cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_length)
            #cv2.imshow("Aruco Detection", cv_image)
           # cv2.waitKey(1)


if __name__ == '__main__':
    try:
        ArucoDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()

