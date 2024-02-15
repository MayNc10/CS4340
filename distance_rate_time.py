#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:36:51 2024

@author: may
"""

distance = float(input("What was the distance traveled? "))
time = float(input("How long did it take to travel that distance? "))
speed = distance / time
print(f"The speed was {speed:.3f}")

distance = float(input("What was the distance traveled? "))
speed = float(input("How fast did it go? "))
time = distance / speed
print(f"The time to travel was {time}")

speed = float(input("How fast did it go? "))
time = float(input("How long did it take to travel that distance? "))
distance = speed * time
print(f"It traveled {distance}")