#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 09:00:31 2024

@author: may
"""

width = float(input("What is the width of the room, in feet?\n"))
length = float(input("What is the length of the room, in feet?\n"))
area_feet = width * length
area_meters = area_feet * (0.3048 ** 2)
print(f"The area in feet is {area_feet:.2f} square feet")
print(f"The area in meters is {area_meters:.2f} square meters")
print("Goodbye")