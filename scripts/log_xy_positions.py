#!/usr/bin/env python3
import rospy
import csv
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

class MotionLogger:
    def __init__(self):
        self.base_pos = None
        self.ee_pos = None

        self.csv_file = open('/home/riri/motion_log.csv', 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['time', 'base_x', 'base_y', 'ee_x', 'ee_y'])

        rospy.Subscriber("/turtlebot/kobuki/odom", Odometry, self.odom_cb)
        rospy.Subscriber("/transform_to_pose_stamped", PoseStamped, self.ee_cb)

    def odom_cb(self, msg):
        self.base_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def ee_cb(self, msg):
        self.ee_pos = (msg.pose.position.x, msg.pose.position.y)
        self.log()

    def log(self):
        if self.base_pos and self.ee_pos:
            timestamp = rospy.Time.now().to_sec()
            row = [timestamp, *self.base_pos, *self.ee_pos]
            self.writer.writerow(row)

    def run(self):
        rospy.spin()
        self.csv_file.close()

if __name__ == "__main__":
    rospy.init_node("motion_logger")
    logger = MotionLogger()
    logger.run()
