#!/usr/bin/env python3
import rospy, numpy as np
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray, Int32, Int32MultiArray
from visualization_msgs.msg import Marker
from tf.transformations import euler_from_quaternion

from src.robot_model  import RobotModel
from src.tasks        import (Position3D, Jointlimits3D, BasePosition3D,
                              BaseOrientation3D, Orientation3D, Configuration3D, JointPosition3D)
from src.utils        import W_DLS
from src.roshelpers   import create_goal_pose, transform_to_pose_stamped
#rostopic pub /goal_set std_msgs/Float64MultiArray "data: [2.0, 0.01, -0.25, 0.0]"


# ---------------------------------------------------------------------------

class TaskPriorityController:
    def __init__(self):
        rospy.init_node("task_priority_node")
        rospy.loginfo("Task Priority Controller Node Initialised")

        # ------------ state -------------------------------------------------
        self.dof           = 6                        # ψ, d, q1-q4
        self.q             = np.zeros((self.dof, 1))
        self.base_pose     = np.zeros(3)
        self.prev_time     = rospy.Time.now().to_sec()
        self.goal          = None
        self.wheel_r       = 0.035
        self.wheel_b       = 0.23
        self.max_velocity = 0.4

        # task containers
        self.all_tasks: list = []
        self.tasks:     list = []
        self.selected_task_indices: list[int] = []    # what /task_server asked for

        # misc
        self.weight  = np.ones(self.dof)
        self.robot   = RobotModel(np.zeros((4, 1)))
        self.rate    = rospy.Rate(30)

        # ------------ pubs --------------------------------------------------
        self.pose_pub  = rospy.Publisher("/transform_to_pose_stamped",
                                         PoseStamped,   queue_size=10)
        self.goal_pub  = rospy.Publisher("/goal", PoseStamped,  queue_size=1)
        self.joint_pub = rospy.Publisher(
            "/turtlebot/swiftpro/joint_velocity_controller/command",
            Float64MultiArray, queue_size=1)
        self.wheel_pub = rospy.Publisher(
            "/turtlebot/kobuki/commands/wheel_velocities",
            Float64MultiArray, queue_size=1)
        self.wheel_js_pub = rospy.Publisher("/velocities", JointState, queue_size=10)

        self.des_marker_pub  = rospy.Publisher("/desired_ee_pos_marker", Marker, queue_size=1)
        self.curr_marker_pub = rospy.Publisher("/current_ee_pos_marker", Marker, queue_size=1)

        # ------------ subs --------------------------------------------------
        rospy.Subscriber("/weight_set", Float64MultiArray, self.weight_cb)
        rospy.Subscriber("/task_server", Int32MultiArray,  self.task_select_cb)
        rospy.Subscriber("/turtlebot/joint_states", JointState, self.joint_state_cb)
        rospy.Subscriber("/turtlebot/kobuki/odom", Odometry, self.odom_cb)
        rospy.Subscriber("/goal_set", Float64MultiArray, self.goal_cb)

    # ---------------------------------------------------------------- weight
    def weight_cb(self, msg):
        self.weight = msg.data

    # ---------------------------------------------------------------- tasks
    def task_select_cb(self, msg):
        indices = list(map(int, msg.data))
        self.selected_task_indices = indices      
        if not self.all_tasks:
            rospy.loginfo("[task_server] Selection cached, waiting for goal")
            return
        self._apply_selection(indices)

    def _apply_selection(self, indices):
        bad = [i for i in indices if i < 0 or i >= len(self.all_tasks)]
        if bad:
            rospy.logerr("Invalid task indices %s (max=%d)", bad, len(self.all_tasks)-1)
            return
        self.tasks = [self.all_tasks[i] for i in indices]
        names = [t.name for t in self.tasks]

    # ---------------------------------------------------------------- goal
    def goal_cb(self, msg):
        if len(msg.data) != 4:
            rospy.logerr("Goal needs 4 floats [x y z yaw]")
            return
        x, y, z, yaw = msg.data
        self.goal = [x, y, z, yaw]

        # publish goal pose (for visualisation)
        self.goal_pub.publish(create_goal_pose([x, y, z], yaw))

        # build full task list
        dx = -0.70
        base_target = np.array([[x],
                                [y]])

        self.all_tasks = [
            Jointlimits3D("Joint lim 1", np.zeros((1,)), [+0.5,+0.55,-0.55,-0.5], 2),  #0
            Jointlimits3D("Joint lim 2", np.zeros((1,)), [+0.5,+0.55,-0.55,-0.5], 3),  #1
            Jointlimits3D("Joint lim 3", np.zeros((1,)), [+0.5,+0.55,-0.55,-0.5], 4),  #2
            Jointlimits3D("Joint lim 4", np.zeros((1,)), [+0.5,+0.55,-0.55,-0.5], 5),  #3
            Position3D   ("EE XYZ",      np.array([[x],[y],[z]]), 6),                   #4
            BaseOrientation3D("Base yaw", np.array([[yaw]]), 1),                       #5
            BasePosition3D("Base XY", base_target, 1),                                 #6
            Orientation3D("EE yaw", np.array([[yaw]]), 6),                             #7
            Configuration3D("Full cfg", np.array([[x],[y],[z],[yaw]]), 6)  ,            #8
            JointPosition3D("Joint1 to front", np.array([[yaw]]), link=2)  #9

        ]

        if self.selected_task_indices:
            self._apply_selection(self.selected_task_indices)
        else:
            rospy.loginfo("Goal received, waiting for /task_server to pick tasks")

    # ------------------------------------------------------- joint + odom -----
    def joint_state_cb(self, msg):
        idx = {"turtlebot/swiftpro/joint1": 2,
               "turtlebot/swiftpro/joint2": 3,
               "turtlebot/swiftpro/joint3": 4,
               "turtlebot/swiftpro/joint4": 5}
        for n, p in zip(msg.name, msg.position):
            if n in idx:
                self.q[idx[n]] = p

    def odom_cb(self, odom):
        q = odom.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        p = odom.pose.pose.position
        self.base_pose = np.array([p.x, p.y, yaw])

    # ---------------------------------------------------------------- solver
    def solve(self):
        now = rospy.Time.now().to_sec()
        dt  = max(now - self.prev_time, 1e-3)
        self.prev_time = now

        P  = np.eye(self.dof)
        dq = np.zeros((self.dof,1))

        self.robot.q = self.q[2:]
        self.robot.update(dq, dt, self.base_pose)

        for task in self.tasks:
            task.update(self.robot)
            if not task.is_active():
                continue
            J   = task.getJacobian()
            Jb  = J @ P
            err = task.getError()
            dq += W_DLS(Jb, 0.1, self.weight) @ (err - J @ dq)
            s = np.max(dq/self.max_velocity)        # Check if velocities exceed the limit
            if s>1 : 
                dq = dq/s
            P  -= np.linalg.pinv(Jb) @ Jb

        self.robot.update(dq, dt, self.base_pose)
        return dq

    # ---------------------------------------------------------------- output
    def send_velocities(self, dq):
        # arm
        self.joint_pub.publish(Float64MultiArray(data=[float(dq[i]) for i in range(2,6)]))
        # wheels
        r, b = self.wheel_r, self.wheel_b
        v_r = (2*dq[1]+dq[0]*b)/(2*r)
        v_l = (2*dq[1]-dq[0]*b)/(2*r)
        self.wheel_pub.publish(Float64MultiArray(data=[float(v_r), float(v_l)]))
        js = JointState(); js.velocity=[float(v_r),float(v_l)]
        js.header.stamp=rospy.Time.now()
        self.wheel_js_pub.publish(js)

    # ---------------------------------------------------------------- run
    def run(self):
        while not rospy.is_shutdown():
            if self.goal is None:
                rospy.logwarn_throttle(5, "Waiting for /goal_set …")
                self.rate.sleep(); continue
            if not self.tasks:
                rospy.logwarn_throttle(5, "No active tasks (send /task_server)")
                self.rate.sleep(); continue

            dq = self.solve()
            self.send_velocities(dq)

            # visualise current & desired EE (if any Position3D task active)
            pos_cur = transform_to_pose_stamped(self.robot.getEETransform())
            self.pose_pub.publish(pos_cur)

            for t in self.tasks:
                if isinstance(t, Position3D):
                    des = t.getDesired().flatten()
                    self._marker(des, self.des_marker_pub, (1,0,0))
            self._marker([pos_cur.pose.position.x,
                          pos_cur.pose.position.y,
                          pos_cur.pose.position.z], self.curr_marker_pub,(0,1,0))
            self.rate.sleep()

    # ---------------------------------------------------------------- utils
    def _marker(self, pos, pub, colour):
        pos = np.array(pos).flatten()
        if pos.size != 3: return
        m = Marker()
        m.header.frame_id="world_ned"; m.header.stamp=rospy.Time.now()
        m.type=Marker.SPHERE; m.action=Marker.ADD
        m.pose.position.x,m.pose.position.y,m.pose.position.z=pos
        m.scale.x=m.scale.y=m.scale.z=0.05
        m.color.r,m.color.g,m.color.b=colour; m.color.a=1.0
        pub.publish(m)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        TaskPriorityController().run()
    except rospy.ROSInterruptException:
        pass
