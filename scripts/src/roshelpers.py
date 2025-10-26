import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion
import tf.transformations as tf
from tf.transformations import quaternion_from_euler


def transform_to_pose_stamped(transform_matrix, frame_id="world_ned"):

    # Create the PoseStamped message structure
    pose_message = PoseStamped()
    
    # Set message metadata
    pose_message.header.frame_id = frame_id
    pose_message.header.stamp = rospy.Time.now()
    
    # Extract and set translation components
    position = transform_matrix[:3, 3]  # Last column is translation
    pose_message.pose.position.x = position[0]
    pose_message.pose.position.y = position[1]
    pose_message.pose.position.z = position[2]
    
    # Convert rotation matrix to quaternion and set orientation
    rotation_quaternion = tf.quaternion_from_matrix(transform_matrix)
    pose_message.pose.orientation.x = rotation_quaternion[0]
    pose_message.pose.orientation.y = rotation_quaternion[1]
    pose_message.pose.orientation.z = rotation_quaternion[2]
    pose_message.pose.orientation.w = rotation_quaternion[3]
    
    return pose_message


def create_goal_pose(position_xyz, yaw_angle, frame_id="world_ned"):
    pose_message = PoseStamped()
    
    # Configure message header
    pose_message.header.frame_id = frame_id
    pose_message.header.stamp = rospy.Time.now()

    # Set position components
    pose_message.pose.position = Point(
        x=position_xyz[0],
        y=position_xyz[1],
        z=position_xyz[2]
    )
    yaw_angle = float(yaw_angle)
    rospy.loginfo(yaw_angle)
    # Convert yaw to quaternion and set orientation
    q = quaternion_from_euler(0.0, 0.0, yaw_angle)  # Only yaw matters
    pose_message.pose.orientation = Quaternion(
        x=q[0],
        y=q[1],
        z=q[2],
        w=q[3]
    )
    return pose_message
