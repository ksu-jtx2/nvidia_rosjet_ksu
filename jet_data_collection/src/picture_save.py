#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import datetime
import os
import glob

bridge = CvBridge()
save_dir = "./data/images"

def image_callback(msg):
	filename = datetime.datetime.now().strftime("/image_%Y_%m_%d_%I_%M_%S_%f.jpg") #Define the filename
	try:
		cv2_img = bridge.imgmsg_to_cv2(msg,"bgr8")
	except CvBridgeError, e:
		print(e)
	else:
		print("saved to: ",save_dir, filename)
		cv2.imwrite(save_dir + filename, cv2_img)

def main():
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	rospy.init_node('image_listener')
	image_topic = "usb_cam/image_raw/"
	rospy.Subscriber(image_topic, Image, image_callback)
	rospy.spin()
	
if __name__ == '__main__':
    main()
