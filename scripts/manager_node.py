#!/usr/bin/python3

import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point, Pose, Quaternion, PoseStamped
from std_msgs.msg import Int32
from tf.transformations import quaternion_from_euler
from intervention.srv import intervention, interventionResponse
from std_msgs.msg import Int32MultiArray 

class ServiceObject():
    def __init__(self):

        self.weight = None
        self.goal = None
        self.last_aruco_pose = None
        self.marker_seen = False

        self.pub_weight = rospy.Publisher('/weight_set', Float64MultiArray,
                                          queue_size=10)
        self.pub_goal = rospy.Publisher('/goal_set', Float64MultiArray, queue_size=10)
        self.pub_aruco = rospy.Publisher('/aruco_position', Point, queue_size=10)

        self.pub_task = rospy.Publisher("/task_server",
                                Int32MultiArray,  # << change
                                queue_size=10)
        rospy.Service('weight_server', intervention, self.weight_service)

        rospy.Service('aruco_server', intervention, self.aruco_service)

        rospy.Service('goal_server', intervention, self.goal_service)

        rospy.Service('task_server', intervention, self.task_service)
        rospy.sleep(1.0)

    def weight_service(self, msg):
        # Extract weight data from the service message
        weight = msg.data
        # Set the parameter for weighted_DLS in the ROS parameter server
        rospy.set_param('weighted_DLS', weight)

        weighted_DLS = Float64MultiArray()
        # Assign the received value to the message
        weighted_DLS.data = weight
        # Publish the weighted DLS
        self.pub_weight.publish(weighted_DLS)
        # Return a service response
        dummy_pose = Pose()  # or fill in actual values if needed
        return interventionResponse(success=True, pose=dummy_pose)

    def task_service(self, req):
        try:
            indices = [int(i) for i in req.data]
        except TypeError:
            indices = [int(req.data)]
        except AttributeError:
            indices = [int(req.index)]

        rospy.loginfo("task_service → indices %s", indices)
        msg          = Int32MultiArray()
        msg.data     = indices
        self.pub_task.publish(msg)
        return interventionResponse()


    def goal_service(self, msg):
        # Extract goal data from the service message
        goal = msg.data

        #Store goal internally so your controller can access it
        self.goal = goal  

        #Log for debugging
        rospy.loginfo("[goal_service] Received goal: %s", goal)

        # Publish to topic
        goal_msg = Float64MultiArray()
        goal_msg.data = goal
        self.pub_goal.publish(goal_msg)

        # Optionally set as parameter if other nodes use it
        rospy.set_param('goal', goal)

        return interventionResponse()


    def aruco_service(self, msg):
        try:
            if rospy.has_param('aruco_pose'):
                pose_data = rospy.get_param('aruco_pose')  # [x, y, z, yaw]

                x, y, z, yaw = pose_data
                qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)

                pose = Pose()
                pose.position.x = x
                pose.position.y = y
                pose.position.z = z
                pose.orientation.x = qx
                pose.orientation.y = qy
                pose.orientation.z = qz
                pose.orientation.w = qw

                self.pub_aruco.publish(pose.position)
                rospy.loginfo("ArUco pose published.")
                return interventionResponse(success=True, pose=pose)
            else:
                rospy.logwarn("No ArUco pose found on parameter server.")
                return interventionResponse(success=False, pose=Pose())
        except Exception as e:
            rospy.logerr(f"Error in aruco_service: {e}")
            return interventionResponse(success=False, pose=Pose())
        
if __name__ == '__main__':
    try:
        rospy.init_node('intervention_service')
        ServiceObject()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
