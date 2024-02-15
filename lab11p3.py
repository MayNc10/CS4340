#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:02:19 2024

@author: may
"""

from math import *

def round_half_up(n):
    multiplier = 10**-1
    return floor(n * multiplier + 0.5) / multiplier

def main():
    num1 = round_half_up(int(input("Enter the first number ")))
    num2 = round_half_up(int(input("Enter the second number ")))
    num3 = round_half_up(int(input("Enter the third number ")))
    print(f"The rounded sum is {int(num1 + num2 + num3)}")
    
if __name__ == "__main__":
    main()