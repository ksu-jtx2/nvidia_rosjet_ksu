#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 21:56:51 2018

@author: vinh
This is a example of using CSV file format in python
This is part of a personal 3 stage process

1: use python to create a csv and output data into a indexable list with the following data values:
    image_name, encoder_left, encoder_right, motor_left, motor_right
2: Create a loop which saves an image once every ~10 frames to the mountable SD card attached to the JTX2
3: Create a ROS subscriber which subscribes to topics from the arduino:
    encoder_left, encoder right, motor_left, motor_right
    
    subscribes to the webcam:
        image every ~10 frames or so, depending on testing
    
    keeps track of all the information via a CSV
    
    & every new training session labels the dataset with the time and date of when the process is run
    
STAGE ONE: Use Python to manipulate CSV file format: https://docs.python.org/3/library/csv.html
This is using Python2.7 to be compatiable with ROS
"""

import csv

with open('persons.csv', 'wb') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['Name', 'Profession'])
    filewriter.writerow(['Derek', 'Software Developer'])
    filewriter.writerow(['Steve', 'Software Developer'])
    filewriter.writerow(['Paul', 'Manager'])