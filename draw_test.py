#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 13:58:27 2021

@author: toytiny
"""
import os
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
rect1=((50,50), (50,25), 180)
rect2=((75,50),(50,25),180)
tarea=25*50
area=25*50
int_pts=cv2.rotatedRectangleIntersection(rect1,rect2)[1]
order_pts = cv2.convexHull(int_pts, returnPoints=True)
int_area  = cv2.contourArea(order_pts)
inter     = int_area * 1.0 / (tarea + area - int_area + 1e-10)  
print(inter)