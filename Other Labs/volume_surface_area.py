#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:31:08 2024

@author: may
"""

PI = 3.1415

radius = float(input("What is the radius of the sphere? "))
sa = 4 * PI * (radius ** 2)
volume = 4/3 * PI * (radius ** 3)
print(f"The surface area of a sphere with radius {radius} is {sa:.2f} and the volume is {volume:.3f}")