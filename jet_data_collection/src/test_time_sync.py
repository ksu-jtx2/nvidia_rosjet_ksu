#!/usr/bin/env python

import rospy
import message_filters
from std_msgs.msg import Int64
from sensor_msgs.msg import Image

def callback(image, left_encoder, right_encoder, left_speed, right_speed):
    print("yes received all")
    print(left_encoder, right_encoder, left_speed, right_speed)
    

def test_time_sub():
    rospy.init_node('test_time_sync')
    webcam_sub = message_filters.Subscriber("usb_cam/image_raw", Image)
    
    left_encoder_sub = message_filters.Subscriber("arduino/encoder_left_value", Int64)
    right_encoder_sub = message_filters.Subscriber("arduino/encoder_right_value", Int64)
    left_speed_sub = message_filters.Subscriber("arduino/motor_left_speed", Int64)
    right_speed_sub = message_filters.Subscriber("arduino/motor_right_speed", Int64)
    
    ts = message_filters.ApproximateTimeSynchronizer([webcam_sub, left_encoder_sub, right_encoder_sub, left_speed_sub, right_speed_sub], 10, 0.1, allow_headerless=True)
    ts.registerCallback(callback)
    rospy.spin()

if __name__ == '__main__':
    test_time_sub()