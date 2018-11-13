#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 00:21:26 2018

#This ROS node requires a few dependicies to run
ROS
Python 2.7
OpenCV 3.4

See this page for datetime formatting: https://docs.python.org/2/library/datetime.html

This has been modified from 'test_time_sync.py" to function on the JTX2
@author: vinh
"""
import rospy
import message_filters
from std_msgs.msg import Int16
from std_msgs.msg import Int64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import csv
import datetime
import glob
import os

filename = datetime.datetime.now().strftime("data_%Y-%m-%d-%H-%M.csv") #Define the filename
filename='/media/nvidia/DB0D-650B/data/'+filename #hardcode the filename to the time of when the recording started for the hour
filename_check=[filename] #change datatype to work for the boolean check for glob.glob
datalist= ['image_name', 'left_encoder', 'right_encoder', 'left_speed', 'right_speed'] #create column labels
global image_folder_date
image_folder_date = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M")

global imagename

global save_dir
save_dir = "/media/nvidia/DB0D-650B/data/images/" #This saves to the external SD card on the JTX2
bridge = CvBridge()

def csv_reader():
    filecheck = glob.glob(filename) #check for existence of file in /media/nvidia/DB0B-650B/data
    #files = os.listdir(os.curdir) #debug
    if (filecheck == filename_check): #boolean check
        #note the 'r' for read
        with open(filename, mode='r') as f: #read first line and make sure it is datalist, then return function to callback
            reader = csv.reader(f)
            row1 = next(reader)
        if (row1 == datalist): 
            return
        else: # if not ask user to fix the csv and exit the recording function
            print('Exception: incorrect data labels. Please correct')
            exit()
    else:
        with open(filename, mode='w') as f: #note the 'w' for write
            writer = csv.writer(f)
            writer.writerow(datalist) #Write in the data labels once
            
def image_callback(image):
    global save_dir
    global imagename
    imagename = datetime.datetime.now().strftime("%Y_%m_%d_%H/image_%Y_%m_%d_%H_%M_%S_%f.jpg") #Define the filename for image
    # This filename has been placed into the function because we want to call a unique name for each jpg that is created
    # So every time that the image callback function is called, a new filename for the new jpg is created
    # Imagename is a global variable because it is used in another function to save the filename to a csv
    
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
    #convert from Int16 to string to save to csv
    left_encoder = str(left_encoder)
    right_encoder = str(right_encoder)
    left_speed = str(left_speed)
    right_speed = str(right_speed)
    
    #for user debug
    print(imagename, left_encoder[6:], right_encoder[6:], left_speed[6:], right_speed[6:])
    
    #save to csv, format to remove ROS formatting by treating the string as an array of elements, ex:
    #data: <number> -> <number>
    with open(filename, mode='a') as f: #note the 'a' flag for append
        writer = csv.writer(f)
        writer.writerow([imagename[1:],left_encoder[6:], right_encoder[6:], left_speed[6:], right_speed[6:]])
    

def test_time_sub():
    global save_dir
    global image_folder_date
    if not os.path.exists(save_dir+image_folder_date): #create save directiory
        os.makedirs(save_dir+image_folder_date)
        
    #run csv reader once
    csv_reader()
    
    rospy.init_node('test_time_sync') #start ROS node
    
    #Start all message filter subscribers
    webcam_sub = message_filters.Subscriber("usb_cam/image_raw", Image)
    left_encoder_sub = message_filters.Subscriber("arduino/encoder_left_value", Int64)
    right_encoder_sub = message_filters.Subscriber("arduino/encoder_right_value", Int64)
    left_speed_sub = message_filters.Subscriber("arduino/speed_left", Int64)
    right_speed_sub = message_filters.Subscriber("arduino/speed_right", Int64)
    
    #Create object to collect all the subscribers and use the object ApproxTimeSync to grab data from roughly same timestamps
    ts = message_filters.ApproximateTimeSynchronizer([webcam_sub, left_encoder_sub, right_encoder_sub, left_speed_sub, right_speed_sub], 10, 0.1, allow_headerless=True)
    #Call callback function when ts receives a new batch of data
    ts.registerCallback(callback)
    rospy.spin() #Use this so ROS doesn't close the node when the function is done running

if __name__ == '__main__':
    test_time_sub()
