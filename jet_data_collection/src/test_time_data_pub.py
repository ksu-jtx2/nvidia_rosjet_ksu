#!/usr/bin/env python

import rospy
from std_msgs.msg import Int64
import random
rospy.init_node('trash_publisher')
global x

def publisher1(publish):
    pub1 = rospy.Publisher('arduino/encoder_left_value', Int64, queue_size = 10)
    x=random.random()
    pub1.publish(x)
    
def publisher2(publish):
    pub2 = rospy.Publisher('arduino/encoder_right_value', Int64, queue_size = 10)
    x=random.random()
    pub2.publish(x)
    
def publisher3(publish):
    pub3 = rospy.Publisher('arduino/motor_left_speed', Int64, queue_size = 10)
    x=random.random()
    pub3.publish(x)
    
def publisher4(publish):
    pub4 = rospy.Publisher('arduino/motor_right_speed', Int64, queue_size = 10)
    x=random.random()
    pub4.publish(x)
    
rospy.Timer(rospy.Duration(.1), publisher1)
rospy.Timer(rospy.Duration(.2), publisher2)
rospy.Timer(rospy.Duration(.15), publisher3)
rospy.Timer(rospy.Duration(.05), publisher4)

rospy.spin()