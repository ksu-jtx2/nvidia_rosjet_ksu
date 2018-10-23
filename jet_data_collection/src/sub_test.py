#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import csv
import datetime
import os

def filename_check(filename_path):
    """
    Get the path to a filename which does not exist by incrementing path.
    """
    if not os.path.exists(filename_path):
        return filename_path
    filename, file_extension = os.path.splittext(filename_path)
    i = 1
    new_filename = "{}-{}{}".format(filename, i, file_extension)
    while os.path.exists(new_filename):
        i+= 1
        new_filename = "{}-{}{}".format(filename, i, file_extension)
        return new_filename
    
    
filename = datetime.datetime.now().strftime("data_%Y-%m-%d.csv") #Define the filename

with open(filename, mode='w') as f:
    reader = csv.reader(f)
    
    writer = csv.writer(f)
    writer.writerow(['str1', 'str2', 'Date']) #Write in the data labels once

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    received = str(data.data)
    
    with open(filename, mode='a') as f:
        writer = csv.writer(f)
        writer.writerow([received.split()])
        
       
       
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("chatter", String, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()