#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 22:20:26 2018

@author: vinh
"""
import csv

#open file

with open('persons.csv', 'rb') as f:
    reader = csv.reader(f)
    
    for row in reader:
        print row