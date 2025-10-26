#!/usr/bin/env python3
import rospy, py_trees, numpy as np
from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, Int32
from std_srvs.srv import SetBool
from intervention.srv import intervention, interventionRequest, interventionResponse
from sensor_msgs.msg import JointState
from py_trees.blackboard import Client as BlackboardClient
import time
import math
import py_trees.display as display

from tf.transformations import euler_from_quaternion


bb = py_trees.blackboard.Blackboard()

# ----------------------------------------------------- Utilities
class WaitForService(py_trees.behaviour.Behaviour):
    def __init__(self, srv_name, timeout=5.0):
        super().__init__(f"wait_for_{srv_name}")
        self.srv_name, self.timeout = srv_name, timeout
        self.bb = self.attach_blackboard_client(name=self.name)

    def update(self):
        try:
            rospy.wait_for_service(self.srv_name, timeout=self.timeout)
            return py_trees.common.Status.SUCCESS
        except rospy.ROSException:
            return py_trees.common.Status.FAILURE

class CallIntervention(py_trees.behaviour.Behaviour):
    def __init__(self, name, srv_name, data):
        super().__init__(name); self.srv_name, self.data = srv_name, data
    def setup(self):
        self.proxy = rospy.ServiceProxy(self.srv_name, intervention)
    def update(self):
        try:
            self.proxy(self.data); return py_trees.common.Status.SUCCESS
        except rospy.ServiceException as e:
            rospy.logerr(f"[{self.name}] {e}"); return py_trees.common.Status.FAILURE
        
#---------------------------------------------------------------- Start Tree

class CheckIfCompleted(py_trees.behaviour.Behaviour):
    def __init__(self, name="check_if_completed"):
        super(CheckIfCompleted, self).__init__(name)
        rospy.loginfo("CheckIfCompleted behavior initialized")
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(key="task_done", access=py_trees.common.Access.READ)

    def update(self):
        try:
            if self.blackboard.task_done:
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.FAILURE
        except KeyError:
            return py_trees.common.Status.FAILURE


class RotateEE(py_trees.behaviour.Behaviour):
    def __init__(self, goal_yaw=1.57, tolerance=0.02):
        super().__init__("RotateEE_front1")

        self.goal_yaw = goal_yaw
        self.tolerance = tolerance
        self.q = np.zeros(6)  # Initialize joint states
        
        # Setup subscribers and service proxies
        self.sub = rospy.Subscriber(
            "/turtlebot/joint_states",
            JointState,
            self.joint_state_cb
        )
        self.task_srv = rospy.ServiceProxy('task_server', intervention)
        self.weight_srv = rospy.ServiceProxy('weight_server', intervention)
        self.goal_srv = rospy.ServiceProxy('goal_server', intervention)
        
        # Initialize blackboard
        self.bb = self.attach_blackboard_client(name=self.name)
        self.bb.register_key(key="ee_point", access=py_trees.common.Access.WRITE)
    
    def joint_state_cb(self, msg):
        idx = {
            "turtlebot/swiftpro/joint1": 2,
            "turtlebot/swiftpro/joint2": 3,
            "turtlebot/swiftpro/joint3": 4,
            "turtlebot/swiftpro/joint4": 5
        }
        for n, p in zip(msg.name, msg.position):
            if n in idx:
                self.q[idx[n]] = p

    def update(self):
        self.task_srv(interventionRequest(data=[0.0, 9.0]))
        # Set weights (base + arm)
        self.weight_srv(interventionRequest(data=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
        # Set goal
        self.bb.ee_point = [0.0, 0.0, 0.0, 1.57]
        self.goal_srv(interventionRequest(data=self.bb.ee_point))

        # Calculate and log error
        current_yaw = float(self.q[2])
        error = abs(float(self.goal_yaw) - current_yaw)
        #rospy.loginfo(f"[RotateEEToFront] Joint1 error: {error:.3f} rad")

        if error < self.tolerance:
            rospy.loginfo("[RotateEEToFront] EE is correctly oriented.")
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING



class MoveToAruco(py_trees.composites.Sequence):
    def __init__(self):
        super().__init__(
            name="move_to_aruco", memory=True
        )

        self.add_child(DetectAruco())
        self.add_child(MovetoPick(goal_key="forward_point"))

# ------------------------------------------1) DETECT ARUCO
class DetectAruco(py_trees.behaviour.Behaviour):
    def __init__(self):
        super().__init__("detect_aruco")
        self.bb = self.attach_blackboard_client(name=self.name)
        self.bb.register_key(key="aruco_pose", access=py_trees.common.Access.WRITE)
        self.bb.register_key(key="forward_point", access=py_trees.common.Access.WRITE)

    def setup(self):
        self.proxy = rospy.ServiceProxy('aruco_server', intervention)

    def update(self):
        try:
            rospy.loginfo("[DetectAruco] Calling aruco_server...")
            resp = self.proxy(interventionRequest(data=[]))
            if resp.success:
                pose = resp.pose
                rospy.loginfo("[DetectAruco] ArUco detected, pose received.")
                #rospy.loginfo(f"[DetectAruco] Pose: {pose}")

                self.bb.aruco_pose = pose
                offset = -0.3  # distance behind the tag
                self.bb.forward_point = [
                        pose.position.x - 0.25,  # assuming 1.6 is the heading
                        pose.position.y,
                        pose.position.z,
                        1.6
                    ]


                #self.bb.forward_point = [0.0,0.0,0.0,1.57]
                return py_trees.common.Status.SUCCESS
            else:
                rospy.logwarn("[DetectAruco] No marker found by aruco_server.")
                return py_trees.common.Status.FAILURE
        except rospy.ServiceException as e:
            rospy.logerr(f"[DetectAruco] Service call failed: {e}")
            return py_trees.common.Status.FAILURE

#---------------------------------2) MOVE TO PICK
class MovetoPick(py_trees.behaviour.Behaviour):
    def __init__(self, goal_key):
        super().__init__(f"drive_base_to_{goal_key}")
        self.goal_key = goal_key
        self.robot_xy = None
        self.threshold = 0.042
        self.sub = rospy.Subscriber("/turtlebot/kobuki/odom", Odometry, self.odom_cb)
        rospy.Subscriber('/transform_to_pose_stamped', PoseStamped, self.ee_callback)

        self.task_srv = rospy.ServiceProxy('task_server', intervention)
        self.w_srv = rospy.ServiceProxy('weight_server', intervention)
        self.goal_srv = rospy.ServiceProxy('goal_server', intervention)

    
        self.bb = self.attach_blackboard_client(name=self.name)
        self.bb.register_key(key=goal_key, access=py_trees.common.Access.READ)
    
    def ee_callback(self, msg):
        self.ee_pose = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        #rospy.loginfo(f"[MoveToPick] EE Pose: {self.ee_pose}")

    def odom_cb(self, msg):
        self.robot_xy = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
         ])
        #rospy.loginfo(f"[DriveBase] Robot position: {self.robot_xy}")
      
    def update(self):

        if self.robot_xy is None:
            return py_trees.common.Status.RUNNING
        self.task_srv(interventionRequest(data=[6.0 , 5.0]))
        self.w_srv(interventionRequest(data=[30.0]*2 + [1.0]*4))
        goal = self.bb.get(self.goal_key)[:2]
        dist = np.linalg.norm(self.robot_xy - np.array(goal))
        #rospy.loginfo(f"[DriveBase] Distance to goal: {dist:.3f}")

        if dist < self.threshold:
            rospy.loginfo("[DriveBase] Goal reached.")
            return py_trees.common.Status.SUCCESS

        try:
            self.goal_srv(self.bb.get(self.goal_key))
            return py_trees.common.Status.RUNNING
        except rospy.ServiceException as e:
            rospy.logerr(f"[DriveBase] Failed to send goal: {e}")
            return py_trees.common.Status.FAILURE
        

# ----------  EE DROP (ADJUST HEIGHT) ----------
# ----------  EE DROP (USE CURRENT EE POSE) ----------
class PickPoint(py_trees.behaviour.Behaviour):

    def __init__(self):
        super().__init__("pick_point")
        self.cmd_sent    = False
        self.target_z    = None
        self.ee_pose     = None
        self.goal = None

        # live EE pose
        rospy.Subscriber(
            "/transform_to_pose_stamped",
            PoseStamped,
            self.ee_cb)

        # service proxies
        self.task_srv   = rospy.ServiceProxy('task_server',   intervention)
        self.weight_srv = rospy.ServiceProxy('weight_server', intervention)
        self.goal_srv   = rospy.ServiceProxy('goal_server',   intervention)
        self.proxy = rospy.ServiceProxy('/turtlebot/swiftpro/vacuum_gripper/set_pump', SetBool)

        self.bb = self.attach_blackboard_client(name=self.name)
        self.bb.register_key("ee_goal_drop", access=py_trees.common.Access.WRITE)

    def ee_cb(self, data):
        self.ee_pose = [data.pose.position.x, data.pose.position.y, data.pose.position.z]
        self.ee_full_pose = data.pose 
    def _issue_goal(self):
        # build a 4-tuple goal [x, y, z, yaw]
        quat = self.ee_full_pose.orientation
        _, _, yaw = euler_from_quaternion(
            [quat.x, quat.y, quat.z, quat.w])

        self.goal = [
            self.ee_pose[0],
            self.ee_pose[1],
            -0.144,
            yaw
        ]
        self.bb.ee_goal_drop = self.goal          # for visibility in rqt_py_trees

    def update(self):
        # need an EE pose before we can do anything
        if self.ee_pose is None:
            return py_trees.common.Status.RUNNING
        
        self.task_srv  (interventionRequest(data=[4.0]))
        self.weight_srv(interventionRequest(data=[30.0, 30.0, 1.0, 1.0,  1.0,  1.0]))

        # send the goal exactly once
        if not self.cmd_sent:
            self._issue_goal()
            self.goal_srv  (interventionRequest(data=self.goal))
            self.cmd_sent = True
            return py_trees.common.Status.RUNNING

        # monitor progress
        err = math.dist(np.array(self.ee_pose), np.array(self.goal[0:3]))

        #rospy.loginfo(f"[DropEndEffector] z-error = {err:.3f} m")

        if err < 0.01:
            rospy.loginfo("[DropEndEffector] Desired height reached.")
            time.sleep(0.5)
            try:
                resp = self.proxy(True)
                if resp.success:
                    rospy.loginfo("[MoveToPlace] Pump stopped.")
                    time.sleep(2)
                else:
                    rospy.logwarn("[MoveToPlace] Pump stop failed.")
            except rospy.ServiceException as e:
                rospy.logerr(f"[MoveToPlace] Failed to stop pump: {e}")
                return py_trees.common.Status.FAILURE
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING
        



class MovetoPlace(py_trees.behaviour.Behaviour):
    def __init__(self):
        super().__init__("PlaceObject")
        self.stage = 0  # 0 = first goal, 1 = second goal
        self.ee_pose = None
        self.goal = None
        self.cmd_sent = False

        rospy.Subscriber("/transform_to_pose_stamped", PoseStamped, self.ee_cb)

        self.task_srv   = rospy.ServiceProxy('task_server',   intervention)
        self.weight_srv = rospy.ServiceProxy('weight_server', intervention)
        self.goal_srv   = rospy.ServiceProxy('goal_server',   intervention)

        self.bb = self.attach_blackboard_client(name=self.name)
        self.bb.register_key("place", access=py_trees.common.Access.WRITE)

    def ee_cb(self, data):
        self.ee_pose = [data.pose.position.x, data.pose.position.y, data.pose.position.z]
        self.ee_full_pose = data.pose
    def odom_cb(self, msg):
        self.robot_xy = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
         ])

    def _issue_goal(self):

        # build a 4-tuple goal [x, y, z, yaw]
        quat = self.ee_full_pose.orientation
        _, _, yaw = euler_from_quaternion(
            [quat.x, quat.y, quat.z, quat.w])

        self.goal = [
            self.ee_pose[0] + 0.05,
            self.ee_pose[1],
            -0.3,
            yaw
        ]
        self.bb.place = self.goal          # for visibility in rqt_py_trees


    def update(self):
        # need an EE pose before we can do anything
        if self.ee_pose is None:
            return py_trees.common.Status.RUNNING
        
        self.task_srv  (interventionRequest(data=[4.0]))
        self.weight_srv(interventionRequest(data=[1000.0, 1000.0, 1.0, 1.0,  1.0,  1.0]))

        # send the goal exactly once
        if not self.cmd_sent:
            self._issue_goal()
            self.goal_srv  (interventionRequest(data=self.goal))
            self.cmd_sent = True
            return py_trees.common.Status.RUNNING

        # monitor progress
        err = math.dist(np.array(self.ee_pose), np.array(self.goal[0:3]))
        #rospy.loginfo(f"[DropEndEffector] z-error = {err:.3f} m")

        if err < 0.02:
            rospy.loginfo("[DropEndEffector] Desired height reached.")
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING

class PlacePoint(py_trees.behaviour.Behaviour):
    def __init__(self):
        super().__init__("MoveToPlace")
        self.stage = 0
        self.robot_state = None
        self.ee_pose = None
        self.goal = None

        rospy.Subscriber("/turtlebot/kobuki/odom", Odometry, self.odom_cb)
        rospy.Subscriber("/transform_to_pose_stamped", PoseStamped, self.ee_callback)

        self.task_srv = rospy.ServiceProxy('task_server', intervention)
        self.weight_srv = rospy.ServiceProxy('weight_server', intervention)
        self.goal_srv = rospy.ServiceProxy('goal_server', intervention)
        self.proxy = rospy.ServiceProxy('/turtlebot/swiftpro/vacuum_gripper/set_pump', SetBool)

        self.bb = self.attach_blackboard_client(name=self.name)
        self.bb.register_key("moveto", access=py_trees.common.Access.WRITE)
     

    def ee_callback(self, msg):
        self.ee_pose = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
    def odom_cb(self, msg):
        self.robot_state = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        ])

    def update(self):
        if self.ee_pose is None and self.robot_state is None:
            return py_trees.common.Status.RUNNING

       
        # Stage 0: Move to (x, y - 0.27)
        if self.stage == 0:
            self.task_srv(interventionRequest(data=[4.0]))
            self.weight_srv(interventionRequest(data=[1000.0, 1000.0, 1.0, 1.0, 1.0, 1.0]))

            self.goal = [self.robot_state[0], self.robot_state[1] - 0.27, -0.365, 0.0]
            self.goal_srv(interventionRequest(data=self.goal))
            self.bb.moveto = self.goal
            self.stage = 1
            return py_trees.common.Status.RUNNING

        if self.stage == 1:
            err = np.linalg.norm(self.ee_pose - np.array(self.goal[0:3]))
            rospy.loginfo(f"[MoveToPlace] Stage 1 error = {err:.3f}")
            if err > 0.023:
                return py_trees.common.Status.RUNNING

            # Proceed to stage 2
            self.task_srv(interventionRequest(data=[6.0, 5.0]))
            self.weight_srv(interventionRequest(data=[1.0, 1.0, 1000.0, 1000.0, 1000.0, 1000.0]))
            print("stage 2 goal")
            self.goal = [self.robot_state[0] + 1.5, -0.01,-0.36, 0.0]
            print("sent goal")
            self.goal_srv(interventionRequest(data=self.goal))
            self.bb.moveto = self.goal
            self.stage = 2
            return py_trees.common.Status.RUNNING
             
        if self.stage == 2:
            err1 = np.linalg.norm(self.robot_state - np.array(self.goal[0:2]))
            rospy.loginfo(f"[MoveToPlace] Stage 2 error = {err1:.3f}")
            if err1 > 0.089:
                return py_trees.common.Status.RUNNING

            # Proceed to final goal
            print("stage 3 goal")
            self.task_srv  (interventionRequest(data=[4.0]))
            self.weight_srv(interventionRequest(data=[1000, 1000, 1.0, 1.0,  1.0,  1.0]))
             
            self.goal = [self.robot_state[0] + 0.25, self.robot_state[1] - 0.175, -0.3, 0.0]
            self.goal_srv(interventionRequest(data=self.goal))
            self.bb.moveto = self.goal
            self.stage = 3
            return py_trees.common.Status.RUNNING

        if self.stage == 3:
            print("last stage")
            err2 = np.linalg.norm(np.array(self.ee_pose) - np.array(self.goal[0:3]))  # recommended
            rospy.loginfo(f"[MoveToPlace] Stage 3 error = {err2:.3f}")
            if err2 > 0.05:
                return py_trees.common.Status.RUNNING
            
            print("stage 4 goal")
            self.task_srv  (interventionRequest(data=[4.0]))
            self.weight_srv(interventionRequest(data=[1000.0, 1000.0, 1.0, 1.0,  1.0,  1.0]))
             
            self.goal = [self.robot_state[0] + 0.4, self.robot_state[1] - 0.017, -0.125, 0.0]
            self.goal_srv(interventionRequest(data=self.goal))
            self.bb.moveto = self.goal
            self.stage = 4
            return py_trees.common.Status.RUNNING

        if self.stage == 4:
            print("enter stage4")
            err3 = np.linalg.norm(np.array(self.ee_pose) - np.array(self.goal[0:3]))  
            rospy.loginfo(f"[MoveToPlace] Stage 4 error = {err3:.3f}")
            if err3 > 0.05:
                return py_trees.common.Status.RUNNING
        try:
            resp = self.proxy(False)
            if resp.success:
                rospy.loginfo("[MoveToPlace] Pump stopped.")
                time.sleep(0.5)
            else:
                rospy.logwarn("[MoveToPlace] Pump stop failed.")
        except rospy.ServiceException as e:
            rospy.logerr(f"[MoveToPlace] Failed to stop pump: {e}")
            return py_trees.common.Status.FAILURE

        return py_trees.common.Status.SUCCESS


class GoHome(py_trees.behaviour.Behaviour):
    def __init__(self):
        super().__init__("Goback")
        self.stage = 0
        self.robot_state = None
        self.ee_pose = None
        self.ee_full_pose = None
        self.goal = None

        rospy.Subscriber("/turtlebot/kobuki/odom", Odometry, self.odom_cb)
        rospy.Subscriber("/transform_to_pose_stamped", PoseStamped, self.ee_callback)

        self.task_srv = rospy.ServiceProxy('task_server', intervention)
        self.weight_srv = rospy.ServiceProxy('weight_server', intervention)
        self.goal_srv = rospy.ServiceProxy('goal_server', intervention)

        self.bb = self.attach_blackboard_client(name=self.name)
        self.bb.register_key("home", access=py_trees.common.Access.WRITE)

    def ee_callback(self, msg):
        self.ee_pose = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.ee_full_pose = msg.pose

    def odom_cb(self, msg):
        self.robot_state = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        ])

    def update(self):
        if self.ee_pose is None or self.robot_state is None:
            return py_trees.common.Status.RUNNING

        if self.stage == 0:
            self.task_srv(interventionRequest(data=[4.0]))
            self.weight_srv(interventionRequest(data=[1000.0, 1000.0, 1.0, 1.0, 1.0, 1.0]))

            quat = self.ee_full_pose.orientation
            _, _, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])

            self.goal = [self.ee_pose[0], self.ee_pose[1], -0.3, yaw]
            self.goal_srv(interventionRequest(data=self.goal))
            self.bb.home = self.goal
            self.stage = 1
            return py_trees.common.Status.RUNNING

        if self.stage == 1:
            err = np.linalg.norm(np.array(self.ee_pose) - np.array(self.goal[0:3]))
            rospy.loginfo(f"[Gohome] Stage 1 error = {err:.3f}")
            if err > 0.023:
                return py_trees.common.Status.RUNNING

            self.task_srv(interventionRequest(data=[6.0, 5.0]))
            self.weight_srv(interventionRequest(data=[1000.0, 1000.0, 1.0, 1.0, 1.0, 1.0]))

            rospy.loginfo("Stage 2: Going home")
            self.goal = [0.0, 0.0, 0.0,0.0]
            self.goal_srv(interventionRequest(data=self.goal))
            self.bb.home = self.goal
            self.stage = 2
            return py_trees.common.Status.RUNNING

        if self.stage == 2:
            err = math.dist(self.goal[:2], self.robot_state)
            rospy.loginfo(f"[Gohome] Home position error = {err:.3f}")
            if err > 0.2:
                return py_trees.common.Status.RUNNING

            rospy.loginfo("[Gohome] Reached home position.")
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING


class SetCompleted(py_trees.behaviour.Behaviour):
    def __init__(self, name="SetCompleted"):
        super(SetCompleted, self).__init__(name)
        rospy.loginfo("SetCompleted behavior initialized")
        self.blackboard = self.attach_blackboard_client(name="SetCompleted")
        self.blackboard.register_key(key="task_done", access=py_trees.common.Access.WRITE)

    def update(self):
        self.blackboard.task_done = True
        self.logger.info("Task marked as completed")
        return py_trees.common.Status.SUCCESS
    


# ----------------------------------------------------Main

if __name__ == "__main__":

    rospy.init_node("aruco_pick_place_tree")
    py_trees.logging.level = py_trees.logging.Level.INFO

    # ---- blackboard global init ----
    global_bb = BlackboardClient(name="Global")
    py_trees.blackboard.Blackboard().set("task_done", False)


    root = py_trees.composites.Selector("Main", memory=True)


    # ---- tree structure ----
    seq = py_trees.composites.Sequence("pick_and_place", memory=True)

    seq.add_children([
        WaitForService('goal_server'),
        WaitForService('task_server'),
        WaitForService('weight_server'),
        RotateEE(),
        MoveToAruco(), 
        PickPoint(),    
        MovetoPlace(),
        PlacePoint(),
        GoHome(),
        SetCompleted()
    ])

    root.add_children([
        CheckIfCompleted(),  # First, check if task is already done
        seq                  # If not, run the pick-and-place sequence
    ])


    root.setup_with_descendants()

    rate = rospy.Rate(5) 
    while not rospy.is_shutdown():
        status = root.tick_once()
        if status == py_trees.common.Status.SUCCESS:
            rospy.loginfo("Behavior tree completed successfully")
            break
        rate.sleep()

    rospy.signal_shutdown("Task completed")
