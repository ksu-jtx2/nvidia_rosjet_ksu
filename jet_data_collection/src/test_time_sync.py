#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 00:21:26 2018

#This ROS node requires a few dependicies to run
ROS
Python 2.7
OpenCV 3.4

This is an example for message filters to be run with:
usb_cam/usb_cam_node
test_time_data_pub.py

@author: vinh
"""
import rospy
import message_filters
from std_msgs.msg import Int64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import csv
import datetime
import glob
import os

filename = datetime.datetime.now().strftime("data_%Y-%m-%d-%H.csv") #Define the filename
filename='./data/'+filename #hardcode the filename to the time of when the recording started for the house
filename_check=[filename]
datalist= ['image_name', 'data1', 'data2', 'data3', 'data4'] #create column labels

global imagename

global save_dir
save_dir = "./data/images/" #Change this to fit your save location for images
bridge = CvBridge()

def csv_reader():
    filecheck = glob.glob(filename) #check for existence of file in ./data
    #files = os.listdir(os.curdir) #debug
    if (filecheck == filename_check): #boolean check
        with open(filename, mode='r') as f: #read first line and make sure it is datalist, then return function to callback
            reader = csv.reader(f)
            row1 = next(reader)
        if (row1 == datalist): 
            return
        else: # if not ask user to fix the csv and exit the recording function
            print('Exception: incorrect data labels. Please correct')
            exit()
    else:
        with open(filename, mode='w') as f:
            writer = csv.writer(f)
            writer.writerow(datalist) #Write in the data labels once
            
def image_callback(image):
    global save_dir
    global imagename
    imagename = datetime.datetime.now().strftime("/image_%Y_%m_%d_%I_%M_%S_%f.jpg") #Define the filename for image
    
    try:
        cv2_img = bridge.imgmsg_to_cv2(image,"bgr8")
    except CvBridgeError, e:
        print(e)
    else:
        #print("saved to: ",save_dir, imagename) #debug
        cv2.imwrite(save_dir + imagename, cv2_img)

def callback(image, left_encoder, right_encoder, left_speed, right_speed):
    global imagename
    image_callback(image)
    #print("yes received all") #debug
    #convert from int64 to string to save to csv
    left_encoder = str(left_encoder)
    right_encoder = str(right_encoder)
    left_speed = str(left_speed)
    right_speed = str(right_speed)
    
    #for user debug
    print(imagename[1:], left_encoder[6:], right_encoder[6:], left_speed[6:], right_speed[6:])
    
    #save to csv
    with open(filename, mode='a') as f:
        writer = csv.writer(f)
        writer.writerow([imagename[1:],left_encoder[6:], right_encoder[6:], left_speed[6:], right_speed[6:]])
    

def test_time_sub():
    global save_dir
    if not os.path.exists(save_dir): #create save directiory
        os.makedirs(save_dir)
        
    #run csv reader once
    csv_reader()
    
    rospy.init_node('test_time_sync') #start ROS node
    
    #Start all message filter subscribers
    webcam_sub = message_filters.Subscriber("usb_cam/image_raw", Image)
    left_encoder_sub = message_filters.Subscriber("arduino/encoder_left_value", Int64)
    right_encoder_sub = message_filters.Subscriber("arduino/encoder_right_value", Int64)
    left_speed_sub = message_filters.Subscriber("arduino/motor_left_speed", Int64)
    right_speed_sub = message_filters.Subscriber("arduino/motor_right_speed", Int64)
    
    #Create object to collect all the subscribers and use the object ApproxTimeSync to grab data from roughly same timestamps
    ts = message_filters.ApproximateTimeSynchronizer([webcam_sub, left_encoder_sub, right_encoder_sub, left_speed_sub, right_speed_sub], 10, 0.1, allow_headerless=True)
    #Call callback function when ts receives a new batch of data
    ts.registerCallback(callback)
    rospy.spin() #Use this so ROS doesn't close the node when the function is done running

if __name__ == '__main__':
    test_time_sub()